out_dir = 'out-rocstories'
eval_interval = 500
eval_iters = 50 
log_interval = 250

always_save_checkpoint = True

dataset = 'rocstories'
gradient_accumulation_steps = 2  
batch_size = 16 
block_size = 320

n_layer = 7  
n_head = 11
n_embd = 385  
dropout = 0.2
bias = True # False
learning_rate = 2.8e-4 

early_stopping = False # True
early_stopping_patience = 10
early_stopping_min_delta = 0.001
metrics_file = 'metrics_t2_m7_augment.csv'

max_iters = 20000 
lr_decay_iters = 20000 
min_lr = 2.8e-5 
warmup_iters = 1500

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

device = 'cuda'
compile = False