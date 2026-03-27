out_dir = 'out-rocstories'
eval_interval = 500
eval_iters = 50 
log_interval = 250

always_save_checkpoint = True

# wandb_log = False
# wandb_project = 'rocstories'
# wandb_run_name = 'baby-gpt-rocstories'

dataset = 'rocstories'
gradient_accumulation_steps = 1 
batch_size = 16 
block_size = 384 

# baseline baby GPT
n_layer = 6  
n_head = 6
n_embd = 384  
dropout = 0.1
bias = False
learning_rate = 3e-4 

early_stopping = True
early_stopping_patience = 10
early_stopping_min_delta = 0.001
metrics_file = 'metrics.csv'

max_iters = 25000 
lr_decay_iters = 25000 
min_lr = 3e-5 
warmup_iters = 1000

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda'
compile = False