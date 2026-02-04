import argparse
import torch

from exp.exp_CoGenCast import Exp_CoGenCast



import random
import numpy as np
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ddp_setup():
    is_ddp = "LOCAL_RANK" in os.environ and "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, world_size, rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return is_ddp, device, local_rank, world_size, rank

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):  
    return rank == 0



fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description="CoGenCast")


parser.add_argument(
    "--task_name",
    type=str,
    default="pretrain",
    help="task name, options:[pretrain, finetune]",
)
parser.add_argument("--downstream_task", type=str, default="forecast", help="downstream task, options:[forecasting, classification]")
parser.add_argument("--is_training", type=int, default=1, help="status")


parser.add_argument(
    "--is_ddp", type=bool, default=False, help="is_ddp"
)



parser.add_argument(
    "--model_id", type=str, default="CoGenCast", help="model id"
)
parser.add_argument(
    "--model", type=str, default="CoGenCast", help="model name"
)
parser.add_argument(
    "--llm_path", type=str, default="/data/ygliu/Downloads/models/test/Qwen3-0.6B", help="llm model path"
)
parser.add_argument(
    "--backbone", type=str, default="Qwen3-0.6B", help="backbone model name"
)

parser.add_argument(
    "--data", type=str, default="ETTh1", help="dataset type"
)



parser.add_argument(
    "--data_list", type=str, default=None,
    help="comma-separated names of subdatasets for cross-domain (e.g. ETTh2,Exchange,Weather)"
)
parser.add_argument(
    "--crossdomain", type=int, default=0, help=""
)
parser.add_argument(
    "--crosspath", type=str, nargs='+',default="", help=""
)


parser.add_argument(
    "--root_path", type=str, default="./datasets", help="root path of the data file"
)
parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")

parser.add_argument('--time_features', type=int, default=6)  

parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
)
parser.add_argument(
    "--target", type=str, default="OT", help="target feature in S or MS task"
)
parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
)
parser.add_argument(
    "--checkpoints",
    type=str,
    default="./outputs/checkpoints/",
    help="location of model fine-tuning checkpoints",
)
parser.add_argument(
    "--pretrain_checkpoints",
    type=str,
    default="./outputs/pretrain_checkpoints/",
    help="location of model pre-training checkpoints",
)
parser.add_argument(
    "--transfer_checkpoints",
    type=str,
    default="ckpt_best.pth",
    help="checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]",
)
parser.add_argument(
    "--load_checkpoints", type=str, default=None, help="location of model checkpoints"
)
parser.add_argument(
    "--select_channels",
    type=float,
    default=1,
    help="select the rate of channels to train",
)
parser.add_argument(
    "--use_norm",
    type=int,
    default=1,
    help="use normalization",
)
parser.add_argument(
    "--accumulation_steps",
    type=int,
    default=1,
    help="number of accumulation steps",
)


parser.add_argument("--seq_len", type=int, default=336, help="input sequence length")
parser.add_argument("--input_len", type=int, default=336, help="input sequence length")
parser.add_argument("--label_len", type=int, default=0, help="start token length")
parser.add_argument(
    "--pred_len", type=int, default=96, help="prediction sequence length"
)
parser.add_argument(
    "--test_pred_len", type=int, default=96, help="test prediction sequence length"
)
parser.add_argument(
    "--seasonal_patterns", type=str, default="Monthly", help="subset for M4"
)


parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
parser.add_argument("--num_kernels", type=int, default=3, help="for Inception")
parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
parser.add_argument("--c_out", type=int, default=7, help="output size")
parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
parser.add_argument("--pt_layers", type=int, default=2, help="num of encoder layers")
parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
parser.add_argument(
    "--moving_avg", type=int, default=25, help="window size of moving average"
)
parser.add_argument("--factor", type=int, default=1, help="attn factor")
parser.add_argument(
    "--distil",
    action="store_false",
    help="whether to use distilling in encoder, using this argument means not using distilling",
    default=True,
)
parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
parser.add_argument(
    "--fc_dropout", type=float, default=0, help="fully connected dropout"
)
parser.add_argument("--head_dropout", type=float, default=0.1, help="head dropout")
parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    help="time features encoding, options:[timeF, fixed, learned]",
)
parser.add_argument("--activation", type=str, default="gelu", help="activation")
parser.add_argument(
    "--output_attention",
    action="store_true",
    help="whether to output attention in ecoder",
)
parser.add_argument(
    "--individual", type=int, default=0, help="individual head; True 1 False 0"
)
parser.add_argument("--pct_start", type=float, default=0.3, help="pct_start")
parser.add_argument("--patch_len", type=int, default=12, help="path length")
parser.add_argument("--stride", type=int, default=12, help="stride")


parser.add_argument(
    "--num_workers", type=int, default=5, help="data loader num workers"
)
parser.add_argument("--itr", type=int, default=1, help="experiments times")
parser.add_argument("--train_epochs", type=int, default=10, help="train epochs")
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size of train input data"
)
parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
parser.add_argument(
    "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
)
parser.add_argument("--des", type=str, default="test", help="exp description")
parser.add_argument("--loss", type=str, default="MSE", help="loss function")
parser.add_argument("--lradj", type=str, default="decay", help="adjust learning rate")
parser.add_argument(
    "--use_amp",
    action="store_true",
    help="use automatic mixed precision training",
    default=False,
)


parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument(
    "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
)
parser.add_argument(
    "--devices", type=str, default="0", help="device ids of multile gpus"
)


parser.add_argument(
    "--time_steps", type=int, default=1000, help="time steps in diffusion"
)
parser.add_argument(
    "--scheduler", type=str, default="cosine", help="scheduler in diffusion"
)

parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay")
parser.add_argument("--mask_ratio", type=float, default=1.0, help="mask ratio")


parser.add_argument("--num_classes", type=int, default=6, help="number of classes")



parser.add_argument('--lm', type=int, default=3, help='average masking length')
parser.add_argument('--positive_nums', type=int, default=3, help='masking series numbers')
parser.add_argument('--rbtp', type=int, default=1, help='0: rebuild the embedding of oral series; 1: rebuild oral series')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
parser.add_argument('--masked_rule', type=str, default='geometric', help='geometric, random, masked tail, masked head')
parser.add_argument('--mask_rate', type=float, default=0.5, help='mask ratio')

parser.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight_decay')
parser.add_argument('--warmup_ratio', type=float, default=0.1 ,help='warmup_ratio')

parser.add_argument('--ispretrain', type=int, default=1, help='TimeDART or Random init')
parser.add_argument('--tuning_manner', type=int, default=1, help='Tuning-free or Generative-Tuning')
parser.add_argument('--iscross', type=int, default=0, help='Cross')
parser.add_argument('--trial', type=int, default=0, help='')

parser.add_argument("--time_scalar",   type=float, default=1000.0,
                    help="Scale t in [0,1] before time embedding, e.g. 100~1000")
parser.add_argument("--num_steps", type=int,   default=5,
                    help="Sampling steps per patch for FM inference")
parser.add_argument('--k_adapt', type=float, default=0.5,
                    help='')
parser.add_argument('--text', type=int, default=0, help='add text')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False



is_ddp, device, local_rank, world_size, rank = ddp_setup()

args.device = device
args.local_rank = local_rank
args.world_size = world_size
args.rank = rank


print("Args in experiment:")
print(args)

Exp_map = {
    "CoGenCast": Exp_CoGenCast
}

Exp = Exp_map[args.model]





if args.task_name == "finetune":
    for ii in range(args.itr):
        
        setting = "{}_{}_{}_{}_il{}_ll{}_pl{}_dm{}_df{}_nh{}_el{}_dl{}_fc{}_dp{}_hdp{}_ep{}_bs{}_lr{}_{}_{}_{}".format(
            args.task_name,
            args.model,
            args.data,
            args.features,
            args.input_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.d_ff,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.factor,
            args.dropout,
            args.head_dropout,
            args.train_epochs,
            args.batch_size,
            args.learning_rate,
            args.ispretrain,
            args.tuning_manner,
            args.iscross,
        )        
                
        exp = Exp(args)  

        if args.tuning_manner == 1  :
            print(">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            if args.downstream_task == "forecast":
                exp.train(setting)
            elif args.downstream_task == "classification":
                exp.cls_train(setting)
                pass

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        if args.downstream_task == "forecast":
            for steps in [1]:
                args.num_steps = steps
                print(f">>>>>> Running inference with num_steps={steps} <<<<<<")
                for pred in [12, 24, 36, 48]:
                    args.pred_len = pred
                    print(f"===== Running inference with pred_len={pred} =====")
                    for _ in range(5):
                        exp.test()
        torch.cuda.empty_cache()

ddp_cleanup()
