import os
import math
import pandas as pd
import matplotlib.pyplot as plt

out_dir = 'out-rocstories'
csv_path = os.path.join(out_dir, 'metrics.csv')

df = pd.read_csv(csv_path)

# sort just in case
df = df.sort_values('iter').reset_index(drop=True)

# Loss curve
plt.figure(figsize=(8, 5))
plt.plot(df['iter'], df['train_loss'], label='Train Loss')
plt.plot(df['iter'], df['val_loss'], label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=300)
plt.close()

# Learning rate curve
plt.figure(figsize=(8, 5))
plt.plot(df['iter'], df['lr'])
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'lr_curve.png'), dpi=300)
plt.close()

# Perplexity curve from val loss
df['val_ppl'] = df['val_loss'].apply(lambda x: math.exp(x))

plt.figure(figsize=(8, 5))
plt.plot(df['iter'], df['val_ppl'])
plt.xlabel('Iteration')
plt.ylabel('Validation Perplexity')
plt.title('Validation Perplexity')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'val_ppl_curve.png'), dpi=300)
plt.close()

best_idx = df['val_loss'].idxmin()
best_iter = df.loc[best_idx, 'iter']
best_val = df.loc[best_idx, 'val_loss']

# Plot validation loss with best checkpoint highlighted
plt.figure(figsize=(8, 5))
plt.plot(df['iter'], df['val_loss'], label='Validation Loss')
plt.scatter(best_iter, best_val, label=f'Best: {best_val:.3f} @ {best_iter}', s=50)
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.title('Validation Loss with Best Checkpoint')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'val_loss_best.png'), dpi=300)
plt.close()

print("Saved:")
print(f" - {os.path.join(out_dir, 'loss_curve.png')}")
print(f" - {os.path.join(out_dir, 'lr_curve.png')}")
print(f" - {os.path.join(out_dir, 'val_ppl_curve.png')}")