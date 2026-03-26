out_dir = 'out-rocstories'
eval_interval = 500
eval_iters = 50 # 25 -> 10
log_interval = 250

always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'baby-gpt-rocstories'

dataset = 'rocstories'
gradient_accumulation_steps = 4 # 1 
batch_size = 16 
block_size = 256

# baseline baby GPT
n_layer = 7 
n_head = 8 
n_embd = 384
dropout = 0.1
bias = False

learning_rate = 2e-4 # 3e-4

early_stopping = True
early_stopping_patience = 6
early_stopping_min_delta = 0.005
metrics_file = 'metrics.csv'

max_iters = 25000 # 6000 -> 10000 [Proposed: 25000]
lr_decay_iters = 25000 # 6000 -> 10000 [Proposed: 25000]
min_lr = 2e-5 # 3e-5
warmup_iters = 1000 # 200 -> 500

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda'
compile = False