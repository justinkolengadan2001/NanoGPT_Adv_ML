out_dir = 'out-rocstories'
eval_interval = 500
eval_iters = 50 
log_interval = 250

always_save_checkpoint = True

dataset = 'rocstories'
gradient_accumulation_steps = 3  
batch_size = 16 
block_size = 624

n_layer = 7  
n_head = 12 
n_embd = 384  
dropout = 0.25
bias = False
learning_rate = 2.75e-4 

early_stopping = False # True
early_stopping_patience = 10
early_stopping_min_delta = 0.001
metrics_file = 'metrics.csv'

max_iters = 20000 
lr_decay_iters = 20000 
min_lr = 2.75e-5 
warmup_iters = 1500

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda'
compile = False