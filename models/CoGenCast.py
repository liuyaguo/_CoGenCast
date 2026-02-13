import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen3Model, AutoConfig

import torch.nn.functional as Func

from layers.CoGenCast_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    TimeStepEmbedding,
    DenoisingPatchDecoder,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding


class FlattenHead(nn.Module):
    def __init__(self, seq_len: int, d_model: int, pred_len: int, dropout: float):
        super().__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.flatten(x)
        x = self.forecast_head(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_len = args.input_len

        self.alpha = torch.nn.Parameter(torch.tensor(0.0))

        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.use_norm = args.use_norm
        self.num_steps = getattr(args, "num_steps", 1)      
        self.k_adapt = getattr(args, "k_adapt", 0.5)        
        self.time_scalar = getattr(args, "time_scalar", 1000.0)

        self.channel_independence = ChannelIndependence()

        
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.patch = Patch(patch_len=self.patch_len, stride=self.stride)
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1

        
        if args.input_len ==336:
            self.hist_emb_96_1 = nn.Linear(96, self.d_model)
            self.hist_emb_96_2 = nn.Linear(96, self.d_model)
            self.hist_emb_48   = nn.Linear(48, self.d_model)
        self.enc_embedding = PatchEmbedding(patch_len=self.patch_len, d_model=self.d_model)
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=self.dropout)

        
        self.sos_token = nn.Parameter(torch.randn(1, 1, self.d_model), requires_grad=True)
        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(sos_token=self.sos_token)

        
        self.encoder = self.__init_encoder(args)

        if self.task_name == "finetune":
            if args.backbone == "Qwen3-0.6B":
                cfg2 = AutoConfig.from_pretrained(args.llm_path, local_files_only=True)
                cfg2._attn_implementation = "eager"
                cfg2.use_cache = False
                self.bid_encoder = Qwen3Model.from_pretrained(
                    args.llm_path,
                    config=cfg2,
                    local_files_only=True,
                )
            else:
                raise ValueError("Unsupported backbone for bidirectional encoder")

        
        self.diffusion = Diffusion(time_steps=args.time_steps, scheduler=args.scheduler)
        self.time_step_embedding = TimeStepEmbedding(d_model=args.d_model)

        
        self.denoising_patch_decoder = DenoisingPatchDecoder(
            d_model=args.d_model,
            num_layers=args.d_layers,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            mask_ratio=args.mask_ratio,
        )
        self.projection = ARFlattenHead(d_model=self.d_model, patch_len=self.patch_len, dropout=args.head_dropout)


    @staticmethod
    def _neg_inf_like(dtype: torch.dtype) -> float:
        
        if dtype in (torch.float16, torch.bfloat16):
            return -1e4
        else:
            return -1e9

    def _make_hist_future_mask(self, B: int, Sx: int, Sy: int, device, dtype):

        S = Sx + Sy
        NEG_INF = self._neg_inf_like(dtype)
        mask = torch.zeros((B, 1, S, S), device=device, dtype=dtype)

        if Sx > 0:
            
            mask[:, :, :Sx, :] = NEG_INF
            
            diag_idx = torch.arange(Sx, device=device)
            mask[:, :, diag_idx, diag_idx] = 0.0

        if Sy > 0:
            
            
            tri = torch.triu(torch.ones((Sy, Sy), device=device, dtype=dtype), diagonal=1)
            mask_future = tri * NEG_INF  
            mask[:, :, Sx:S, Sx:S] = mask_future  

        
        return mask

    def _encode_future_with_frozen_history(self, hist_emb: torch.Tensor, fut_emb: torch.Tensor):

        B, Sx, D = hist_emb.shape
        Sy = fut_emb.shape[1]
        concat = torch.cat([hist_emb, fut_emb], dim=1)          

        enc_dtype = next(self.encoder.parameters()).dtype
        if concat.dtype != enc_dtype:
            concat = concat.to(dtype=enc_dtype)

        mask = self._make_hist_future_mask(B, Sx, Sy, concat.device, enc_dtype)

        out = self.encoder(
            inputs_embeds=concat,
            attention_mask={"full_attention": mask},
            return_dict=True,
        )
        return out.last_hidden_state[:, Sx:, :]

    

    def __init_encoder(self, args):

        if args.backbone == "Qwen3-0.6B":
            cfg = AutoConfig.from_pretrained(args.llm_path, local_files_only=True)
            cfg._attn_implementation = "eager"  
            cfg.use_cache = False
            enc = Qwen3Model.from_pretrained(
                args.llm_path,
                config=cfg,
                local_files_only=True,
            )
        else:
            raise ValueError(f"Backbone {args.backbone} not supported")
        return enc


    def forecast_train(self, x, y,text_emb):
        B, L, F = x.size()
        device = x.device
        P = self.patch_len
        
        x_means = x.mean(dim=1, keepdim=True).detach()
        x_center = x - x_means
        x_stdevs = torch.sqrt(x_center.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_n = x_center / x_stdevs
        y_n = (y - x_means) / x_stdevs

        
        x_ci = self.channel_independence(x_n)    

        BFx, Lx, _ = x_ci.shape
        stride = self.stride
 
        x_p = self.patch(x_ci)                      
        x_emb = self.enc_embedding(x_p)             
        BF2, Sx, D = x_emb.shape


        B_text, L_text, D_text = text_emb.shape
        assert B_text == B, f"text_emb batch={B_text} 与 x batch={B} 不一致"

        Sx_num = Sx
   
        S_text_target = max(1, min(L_text, 1))


        text_emb_t = text_emb.transpose(1, 2)

        
        text_pooled = Func.adaptive_avg_pool1d(
            text_emb_t,
            output_size=S_text_target
        )                                            
    
        text_tokens = text_pooled.transpose(1, 2)    

        if D_text != D:
            text_tokens = self.text_proj(text_tokens)    
            

        text_tokens_bf = text_tokens.repeat_interleave(F, dim=0)   
        lea = torch.sigmoid(self.alpha)  
        text_tokens_bf = lea * text_tokens_bf +(1-lea)  * x_emb[:, 0:1, :] 
 
        x_emb = torch.cat([text_tokens_bf, x_emb], dim=1)          
        
        S_text = S_text_target


        if self.args.backbone == "Transformer":
            x_out = self.bid_encoder(x_emb, is_mask=False)
        elif self.args.backbone == "Qwen2.5-0.5B":
            BFx, Sx, _ = x_emb.shape
            bid_dtype = next(self.bid_encoder.parameters()).dtype
            if x_emb.dtype != bid_dtype:
                x_emb = x_emb.to(dtype=bid_dtype)
            mask_bi = torch.zeros((BFx, 1, Sx, Sx), device=x_emb.device, dtype=bid_dtype)
            out_x = self.bid_encoder(
                inputs_embeds=x_emb,
                attention_mask={"full_attention": mask_bi}, 
                output_hidden_states=True,
                return_dict=True,
            )
            x_out = out_x.hidden_states[-1]  
        elif self.args.backbone == "Qwen3-0.6B":
            BFx, Sx, _ = x_emb.shape
            bid_dtype = next(self.bid_encoder.parameters()).dtype
            if x_emb.dtype != bid_dtype:
                x_emb = x_emb.to(dtype=bid_dtype)
            mask_bi = torch.zeros((BFx, 1, Sx, Sx), device=x_emb.device, dtype=bid_dtype)
            out_x = self.bid_encoder(
                inputs_embeds=x_emb,
                attention_mask={"full_attention": mask_bi},  
                return_dict=True,
            )
            x_out = out_x.last_hidden_state  

        else:
            raise ValueError("Unsupported backbone")

 
        y_ci = self.channel_independence(y_n)    
        y_patch = self.patch(y_ci)               
        BFy, Sy, P = y_patch.shape

        y_emb = self.enc_embedding(y_patch)      
        y_bias = self.add_sos_token_and_drop_last(y_emb)
        y_bias = self.positional_encoding(y_bias)  


        y_out = self._encode_future_with_frozen_history(x_out, y_bias)  

        t = torch.rand(BFy, device=device)
        r = torch.rand(BFy, device=device) * t
        t_ = t.view(BFy, 1, 1).repeat(1, Sy, 1)
        r_ = r.view(BFy, 1, 1).repeat(1, Sy, 1)

        e = torch.randn_like(y_patch)
        z = (1.0 - t_) * y_patch + t_ * e          
        v = e - y_patch                             

        def model_u(z_in, t_in, r_in):
            if t_in.dim() == 1:
                t_br = t_in.view(BFy, 1).repeat(1, Sy)  
                r_br = r_in.view(BFy, 1).repeat(1, Sy)
            else:
                t_br, r_br = t_in, r_in

            z_emb = self.enc_embedding(z_in)       
            T = int(self.diffusion.time_steps)
            t_idx = torch.clamp((t_br * (T - 1)).long(), 0, T - 1)  

            t_embed = self.time_step_embedding(t_idx)               
            q = self.positional_encoding(z_emb + t_embed)

            h = self.denoising_patch_decoder(
                query=q, key=y_out, value=y_out,
                is_tgt_mask=True, is_src_mask=True,
            )                                        

            h_full = h.view(B, F, Sy, self.d_model)     
            u_seq = self.projection(h_full)             
            u_ci = self.channel_independence(u_seq)     
            u_tok = self.patch(u_ci)                    
            return u_tok

        from torch.autograd.functional import jvp

        t_vec = t.clone().requires_grad_(True)
        r_vec = r.clone().requires_grad_(True)

        def wrapped(z_in, t_scalar, r_scalar):
            return model_u(z_in, t_scalar, r_scalar)

        u_pred, du_dt = jvp(
            lambda z_arg, t_arg, r_arg: wrapped(z_arg, t_arg, r_arg),
            inputs=(z, t_vec, r_vec),
            v=(torch.zeros_like(z), torch.ones_like(t_vec), torch.zeros_like(r_vec)),
            create_graph=self.training,
        )

        if du_dt.ndim == 0:
            du_dt = du_dt.view(1).expand(BFy * Sy)
        if du_dt.ndim == 1:
            if du_dt.numel() == BFy:
                du_dt = du_dt.view(BFy, 1, 1).expand(BFy, Sy, P)
            elif du_dt.numel() == BFy * Sy:
                du_dt = du_dt.view(BFy, Sy, 1).expand(BFy, Sy, P)
            else:
                raise RuntimeError(f"Unexpected du_dt.numel()={du_dt.numel()} (expected {BFy} or {BFy*Sy})")
        elif du_dt.ndim == 3:
            if du_dt.shape != (BFy, Sy, P):
                raise RuntimeError(f"Unexpected du_dt.shape={tuple(du_dt.shape)}, expected ({BFy},{Sy},{P})")
        else:
            raise RuntimeError(f"Unexpected du_dt.ndim={du_dt.ndim}")

        t_minus_r = (t_ - r_)                                  
        u_tgt = v - t_minus_r * du_dt                          
        loss = torch.mean((u_pred - u_tgt.detach()) ** 2)
        return loss

    @torch.no_grad()
    def forecasting_test(self, x, max_len,text_emb):
        device = x.device
        B, L, F = x.size()
        assert L == self.input_len, f"Expected input_len={self.input_len}, got {L}"

        P = self.patch_len
        K = max(1, int(getattr(self.args, "num_steps", 1)))  
        preds_all = []
        remain = max_len
        step_size = 12  

        def _gen_chunk_from_hist(hist_real, chunk_len):
            
            if self.use_norm:
                x_means = hist_real.mean(dim=1, keepdim=True).detach()
                x_center = hist_real - x_means
                x_stdevs = torch.sqrt(x_center.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
                hist_n = x_center / x_stdevs
            else:
                hist_n = hist_real
                x_means = torch.zeros(B, 1, F, device=device, dtype=hist_real.dtype)
                x_stdevs = torch.ones(B, 1, F, device=device, dtype=hist_real.dtype)

        
            x_ci = self.channel_independence(hist_n)

            BFx, Lx, _ = x_ci.shape
            stride = self.stride

            x_p = self.patch(x_ci)                      
            x_emb = self.enc_embedding(x_p)             
            BF2, Sx, D = x_emb.shape


            B_text, L_text, D_text = text_emb.shape
            assert B_text == B, f"text_emb batch={B_text} 与 x batch={B} 不一致"

            Sx_num = Sx
         
            S_text_target = max(1, min(L_text, 1))

      
            text_emb_t = text_emb.transpose(1, 2)

            
            text_pooled = Func.adaptive_avg_pool1d(
                text_emb_t,
                output_size=S_text_target
            )                                            
        
            
            text_tokens = text_pooled.transpose(1, 2)    

            
            if D_text != D:
           
                text_tokens = self.text_proj(text_tokens)    
                
         
            text_tokens_bf = text_tokens.repeat_interleave(F, dim=0)   
            lea = torch.sigmoid(self.alpha)  
            text_tokens_bf = lea * text_tokens_bf +(1-lea)  * x_emb[:, 0:1, :] 
            x_emb = torch.cat([text_tokens_bf, x_emb], dim=1)          
        
            S_text = S_text_target


            if self.args.backbone == "Transformer":
                x_out = self.bid_encoder(x_emb, is_mask=False)
            elif self.args.backbone == "Qwen2.5-0.5B":
                BFx, Sx, _ = x_emb.shape
                bid_dtype = next(self.bid_encoder.parameters()).dtype
                if x_emb.dtype != bid_dtype:
                    x_emb = x_emb.to(dtype=bid_dtype)
                mask_bi = torch.zeros((BFx, 1, Sx, Sx), device=device, dtype=bid_dtype)
                out_x = self.bid_encoder(
                    inputs_embeds=x_emb,
                    attention_mask={"full_attention": mask_bi},
                    output_hidden_states=True,
                    return_dict=True,
                )
                x_out = out_x.hidden_states[-1]  
            elif self.args.backbone == "Qwen3-0.6B":
                BFx, Sx, _ = x_emb.shape
                bid_dtype = next(self.bid_encoder.parameters()).dtype
                if x_emb.dtype != bid_dtype:
                    x_emb = x_emb.to(dtype=bid_dtype)
                mask_bi = torch.zeros((BFx, 1, Sx, Sx), device=device, dtype=bid_dtype)
                out_x = self.bid_encoder(
                    inputs_embeds=x_emb,
                    attention_mask={"full_attention": mask_bi},
                    return_dict=True,
                )
                x_out = out_x.last_hidden_state  
                

            else:
                raise ValueError("Unsupported backbone")

      
            num_patches_needed = (chunk_len + P - 1) // P
            t_grid = torch.linspace(1.0, 0.0, steps=K + 1, device=device)

            y_ctx_tokens = []
            preds_norm_tok = []

            def model_u_tokens(z_tokens, t_scalar):
                BFc, Sc, Pc = z_tokens.shape
                z_emb = self.enc_embedding(z_tokens)

                T = int(self.diffusion.time_steps)
                t_idx = torch.clamp((t_scalar * (T - 1)).long(), 0, T - 1)
                t_grid_sr = t_idx.view(1).repeat(BFc * Sc).view(BFc, Sc)
                t_embed = self.time_step_embedding(t_grid_sr)
                q = self.positional_encoding(z_emb + t_embed)

                
                y_bias = self.add_sos_token_and_drop_last(z_emb)
                y_bias = self.positional_encoding(y_bias)
                y_out_step = self._encode_future_with_frozen_history(x_out, y_bias)  

                h = self.denoising_patch_decoder(
                    query=q, key=y_out_step, value=y_out_step,
                    is_tgt_mask=True, is_src_mask=True,
                )

                h_full = h.view(B, F, -1, self.d_model)
                u_seq = self.projection(h_full)         
                u_ci = self.channel_independence(u_seq) 
                u_tok = self.patch(u_ci)                
                return u_tok

            for _ in range(num_patches_needed):
                cur_tok = torch.randn(B * F, 1, P, device=device)
                seq_tok = torch.cat(y_ctx_tokens + [cur_tok], dim=1) if y_ctx_tokens else cur_tok

                for s in range(K):
                    t = t_grid[s]
                    r = t_grid[s + 1]
                    u_tok = model_u_tokens(seq_tok, t)    
                    last = u_tok[:, -1:, :]
                    x_next = seq_tok[:, -1:, :] - (t - r) * last
                    seq_tok = x_next if seq_tok.size(1) == 1 else torch.cat([seq_tok[:, :-1, :], x_next], dim=1)

                preds_norm_tok.append(seq_tok[:, -1:, :])
                y_ctx_tokens.append(seq_tok[:, -1:, :])

            preds_tok = torch.cat(preds_norm_tok, dim=1)
            pred_seq_norm = preds_tok.reshape(B, F, -1).transpose(1, 2).contiguous()
            predictions_norm = pred_seq_norm[:, :chunk_len, :]

            if self.use_norm:
                predictions_real = predictions_norm * x_stdevs[:, 0, :].unsqueeze(1) + x_means[:, 0, :].unsqueeze(1)
            else:
                predictions_real = predictions_norm
            return predictions_real

        hist = x.clone()  
        while remain > 0:
            cur = min(step_size, remain)  
            pred_chunk = _gen_chunk_from_hist(hist, cur)
            preds_all.append(pred_chunk)
            remain -= cur
            hist = torch.cat([hist[:, cur:, :], pred_chunk], dim=1)

        return torch.cat(preds_all, dim=1)

    
    def forward(self, x, y=None, text_emb=None):
        if self.task_name == "pretrain":
            return self.pretrain(x)
        elif self.task_name == "finetune":
            if y is not None:
                return self.forecast_train(x, y,text_emb=text_emb)
            else:
                return self.forecasting_test(x, self.pred_len,text_emb=text_emb)
        else:
            raise ValueError(f"Task name {self.task_name} not supported")
