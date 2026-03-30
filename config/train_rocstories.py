out_dir = 'out-rocstories'
eval_interval = 500
eval_iters = 50 
log_interval = 250

always_save_checkpoint = True

dataset = 'rocstories'
gradient_accumulation_steps = 2  
batch_size = 16 
block_size = 512

n_layer = 7  
n_head = 8 
n_embd = 384  
dropout = 0.125
bias = False
learning_rate = 3.25e-4 

early_stopping = False # True
early_stopping_patience = 10
early_stopping_min_delta = 0.001
metrics_file = 'metrics.csv'

max_iters = 20000 
lr_decay_iters = 20000 
min_lr = 3.25e-5 
warmup_iters = 1500

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda'
compile = False