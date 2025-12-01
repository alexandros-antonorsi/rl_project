import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df1 = pd.read_csv(f'logs/cartpole_dqn_25.csv', index_col='episode_num')
df2 = pd.read_csv(f'logs/cartpole_dqn_42.csv', index_col='episode_num')
df3 = pd.read_csv(f'logs/cartpole_dqn_69.csv', index_col='episode_num')
avg1 = df1['total_reward'].rolling(10).mean()
avg2 = df2['total_reward'].rolling(10).mean()
avg3 = df3['total_reward'].rolling(10).mean()

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10,5))
fig.suptitle("Rewards per episode")
ax1.plot(df1.index, df1['total_reward'], color=(0.059,0.431,0.141), linewidth=1, zorder=0, alpha=0.5)
ax1.plot(df1.index, avg1, alpha=0.8, color=(0.431,0.133,0.059), linewidth=2, label="10-episode rolling reward avg.", zorder=10)
ax2.plot(df2.index, df2['total_reward'], color=(0.059,0.431,0.141), linewidth=1, zorder=0, alpha=0.5)
ax2.plot(df2.index, avg2, alpha=0.8, color=(0.431,0.133,0.059), linewidth=2, zorder=10)
ax3.plot(df3.index, df3['total_reward'], color=(0.059,0.431,0.141), linewidth=1, zorder=0, alpha=0.5)
ax3.plot(df3.index, avg3, alpha=0.8, color=(0.431,0.133,0.059), linewidth=2, zorder=10)
ax1.set_title('seed=25')
ax2.set_title('seed=42')
ax3.set_title('seed=69')
fig.supxlabel("Episode")
fig.supylabel("Reward")
fig.legend()
fig.tight_layout()

plt.savefig(f"plots/plot_cartpole_dqn_25-42-69.png")

print(f"Saved figure to plots/rewards_cartpole_dqn_25-42-69.png")



