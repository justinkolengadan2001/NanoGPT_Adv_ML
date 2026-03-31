import os
import math
import pandas as pd
import matplotlib.pyplot as plt

out_dir = 'out-rocstories'
csv_path = os.path.join(out_dir, 'metrics.csv')

df = pd.read_csv(csv_path)

# sort just in case
df = df.sort_values('iter').reset_index(drop=True)

best_idx = df['val_loss'].idxmin()
best_iter = df.loc[best_idx, 'iter']
best_val = df.loc[best_idx, 'val_loss']

# Loss curve
plt.figure(figsize=(12, 8))
plt.plot(df['iter'], df['train_loss'], label='Train Loss')
plt.plot(df['iter'], df['val_loss'], label='Validation Loss')
plt.scatter(best_iter, best_val, label=f'Best: {best_val:.3f} @ {best_iter}', s=80)

plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title('Training and Validation Loss', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.legend(fontsize=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=200)
plt.close()

# Learning rate curve
plt.figure(figsize=(12, 8))
plt.plot(df['iter'], df['lr'])

plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Learning Rate', fontsize=20)
plt.title('Learning Rate Schedule', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'lr_curve.png'), dpi=200)
plt.close()

# Perplexity curve from val loss
df['val_ppl'] = df['val_loss'].apply(lambda x: math.exp(x))

plt.figure(figsize=(12, 8))
plt.plot(df['iter'][1:], df['val_ppl'][1:])  # skip the first point if it's inf

plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Validation Perplexity', fontsize=20)
plt.title('Validation Perplexity', fontsize=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'val_ppl_curve.png'), dpi=200)
plt.close()

# # Plot validation loss with best checkpoint highlighted
# plt.figure(figsize=(12, 8))
# plt.plot(df['iter'], df['val_loss'], label='Validation Loss')
# plt.scatter(best_iter, best_val, label=f'Best: {best_val:.3f} @ {best_iter}', s=80)

# plt.xlabel('Iteration', fontsize=18)
# plt.ylabel('Validation Loss', fontsize=18)
# plt.title('Validation Loss with Best Checkpoint', fontsize=20)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)

# plt.legend(fontsize=14)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(out_dir, 'val_loss_best.png'), dpi=1000)
# plt.close()

print("Saved:")
print(f" - {os.path.join(out_dir, 'loss_curve.png')}")
print(f" - {os.path.join(out_dir, 'lr_curve.png')}")
print(f" - {os.path.join(out_dir, 'val_ppl_curve.png')}")