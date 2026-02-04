import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masking import generate_causal_mask, generate_self_only_mask, generate_partial_mask


class ChannelIndependence(nn.Module):
    def __init__(
        self,
    ):
        super(ChannelIndependence, self).__init__()

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        _, input_len, _ = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, input_len, 1)
        return x


class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super(AddSosTokenAndDropLast, self).__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  
        x = x[:, :-1, :]  
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )

        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True):
        
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return x























































        




class Diffusion(nn.Module):
    def __init__(self, time_steps: int, scheduler: str = "cosine"):
        super().__init__()
        self.time_steps = int(time_steps)

        
        if scheduler == "cosine":
            betas = self._cosine_beta_schedule()
        elif scheduler == "linear":
            betas = self._linear_beta_schedule()
        else:
            raise ValueError(f"Invalid scheduler: scheduler={scheduler}")

        alpha = 1.0 - betas
        gamma = torch.cumprod(alpha, dim=0)

        
        self.register_buffer("betas", betas)   
        self.register_buffer("alpha", alpha)   
        self.register_buffer("gamma", gamma)   

    @torch.no_grad()
    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)  
        alphas_cumprod = torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0, 0.999).float()

    @torch.no_grad()
    def _linear_beta_schedule(self, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, self.time_steps).float()

    @torch.no_grad()
    def sample_time_steps(self, shape, device) -> torch.Tensor:
        
        return torch.randint(0, self.time_steps, shape, device=device)

    def noise(self, x: torch.Tensor, t: torch.Tensor):
        """
        x: [BF, S, P]; t: [BF, S] (long)
        返回 noisy_x, noise；都与 x 同 dtype/device
        """
        noise = torch.randn_like(x)
        
        gamma_t = self.gamma[t].to(dtype=x.dtype)              
        gamma_t = gamma_t.unsqueeze(-1)                        
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1.0 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x: torch.Tensor):
        """
        x: [BF, S, P]
        这里示例用“全局统一时间步”（同你原来的做法），也可以改为逐 token 采样。
        """
        BF, S, _ = x.shape
        
        t_g = torch.randint(0, self.time_steps, (1,), device=x.device)  
        t = t_g.expand(BF, S)                                           
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t


class TimeStepEmbedding(nn.Module):
    def __init__(self, d_model, max_steps=1000):
        super().__init__()
        self.d_model = d_model
        self.max_steps = 1000

    def forward(self, t):
        """
        :param t: [batch_size] or [batch_size, seq_len], dtype=torch.long
        :return: [batch_size, seq_len, d_model] or [batch_size, d_model] if seq_len=1
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)  
        device = t.device
        half_dim = self.d_model // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)  
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)      
        if emb.size(-1) < self.d_model:
            
            pad = self.d_model - emb.size(-1)
            emb = torch.cat([emb, torch.zeros(t.size(0), t.size(1), pad, device=device)], dim=-1)
        return emb

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerDecoderBlock, self).__init__()


        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)

        
        
        
        

        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value,tgt_mask, src_mask):
        """
        :param query: [batch_size * num_features, seq_len, d_model]
        :param key: [batch_size * num_features, seq_len, d_model]
        :param value: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """

        
        
        
        
        
        
        

        
        residual=query
        query=self.norm1(query)
        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query =residual + self.dropout(attn_output)


        
        residual=query
        query=self.norm2(query)
        attn_output, _ = self.encoder_attention(query, key, key, attn_mask=src_mask)
        query = residual + self.dropout(attn_output)

        
        
        
        
        
        
        
        residual=query
        query=self.norm3(query)
        ff_output = self.ff(query)
        x = residual + self.dropout(ff_output)
        
        
        

        
        
        

        
        
        
        return x


class DenoisingPatchDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        mask_ratio: float,
    ):
        super(DenoisingPatchDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.mask_ratio = mask_ratio

    def forward(self, query, key, value, is_tgt_mask=True, is_src_mask=True):
        seq_len=query.size(1)

        tgt_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_tgt_mask else None
        )
        src_mask = (
            generate_partial_mask(seq_len, self.mask_ratio).to(query.device) if is_src_mask else None
        )
        for layer in self.layers:
            query = layer(query, key, value,tgt_mask, src_mask)
        x = self.norm(query)
        return x


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(self.receptive_field - 1),  
            dilation=dilation,
            groups=groups
        )
        
    def forward(self, x):
        out = self.conv(x)
        
        return out[:, :, :x.size(2)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class CausalTCN(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, kernel_size=3):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        
        
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        
        
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * (depth - 1) + [output_dims],
            kernel_size=kernel_size
        )
        
    def forward(self, x):
        
        
        
        x = self.input_fc(x)
        
        
        x = x.transpose(1, 2)
        
        
        x = self.feature_extractor(x)  
        
        
        x = x.transpose(1, 2)
        
        return x


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):   
        """
        :param x: [batch_size, seq_len, input_dims]
        :return: [batch_size, seq_len, output_dims]
        """
        x = x.transpose(1, 2)
        return self.net(x).transpose(1, 2)


class ClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(ClsHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * d_model, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


class OldClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(OldClsHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(torch.max(x, dim=1)[0])


class ClsEmbedding(nn.Module):
    def __init__(self, num_features, d_model, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_features, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.conv(x).transpose(1, 2) 


class ClsFlattenHead(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, num_features, dropout):
        super(ClsFlattenHead, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len * num_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  
        x = self.dropout(x)  
        x = self.forecast_head(x)  
        return x.reshape(x.size(0), self.pred_len, self.num_features)


class ARFlattenHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_len: int,
        dropout: float,
    ):
        super(ARFlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, seq_len * patch_len, num_features]
        """
        x = self.forecast_head(x)  
        x = self.dropout(x)  
        x = self.flatten(x)  
        x = x.permute(0, 2, 1)  
        return x