out_dir = 'out-rocstories'
eval_interval = 1000
eval_iters = 25
log_interval = 100

always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'baby-gpt-rocstories'

dataset = 'rocstories'
gradient_accumulation_steps = 1
batch_size = 16
block_size = 384

# baseline baby GPT
n_layer = 7 # 6
n_head = 8 # 6
n_embd = 384
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 8000 # 6000
lr_decay_iters = 8000 # 6000
min_lr = 3e-5
warmup_iters = 500 # 200

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda'
compile = False