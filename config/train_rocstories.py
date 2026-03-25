out_dir = 'out-rocstories'
eval_interval = 250
eval_iters = 100
log_interval = 10

always_save_checkpoint = True

wandb_log = False
wandb_project = 'rocstories'
wandb_run_name = 'baby-gpt-rocstories'

dataset = 'rocstories'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256

# baseline baby GPT
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 6000
lr_decay_iters = 6000
min_lr = 3e-5
warmup_iters = 200

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda'
compile = False