"""Generate performance comparison plots for README."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from benchmarks
# Config: B=1, H_q=32, H_kv=8, D=128, num_sink=4, window_size=4096
# All on H200

# Context lengths
N = [512, 1024, 2048, 4096, 8192, 16384, 32768]

# Forward pass latency (ms)
sink_fa_ms = [0.07, 0.12, 0.27, 0.76, 1.88, 4.18, 8.77]
eager_ms = [0.10, 0.29, 1.07, 4.70, 20.20, None, None]  # OOM at 16K+
fa2_ms = [0.04, 0.07, 0.19, 0.52, 1.75, 6.63, 25.83]

# Forward pass memory (MB)
sink_fa_mem = [4, 9, 17, 34, 68, 136, 273]
eager_mem = [38, 144, 558, 2198, 8724, None, None]  # OOM at 16K+
fa2_mem = [4, 9, 17, 34, 68, 136, 273]

# Training (fwd+bwd) latency (ms)
N_train = [512, 1024, 2048, 4096, 8192, 16384]
sink_fa_train_ms = [0.41, 0.47, 0.97, 2.81, 7.28, 16.39]
fa2_train_ms = [0.25, 0.32, 0.74, 2.09, 6.89, 24.63]

# Training memory (MB)
sink_fa_train_mem = [34, 67, 135, 269, 538, 1076]
fa2_train_mem = [42, 84, 168, 337, 673, 1346]

# Colors
SINK_COLOR = '#2563EB'   # bright blue
FA2_COLOR = '#9CA3AF'    # gray
EAGER_COLOR = '#EF4444'  # red
SINK_COLOR_LIGHT = '#93C5FD'

plt.rcParams.update({
    'font.size': 13,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

# ============================================================
# Plot 1: Forward Pass Latency
# ============================================================
ax1 = axes[0]

# Eager attention (with OOM markers)
eager_valid_n = [n for n, m in zip(N, eager_ms) if m is not None]
eager_valid_ms = [m for m in eager_ms if m is not None]
ax1.plot(eager_valid_n, eager_valid_ms, 'o--', color=EAGER_COLOR, linewidth=2,
         markersize=8, label='Eager (materialized)', zorder=2)
# OOM markers
for n, m in zip(N, eager_ms):
    if m is None:
        ax1.annotate('OOM', (n, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 1 else 30),
                     fontsize=10, color=EAGER_COLOR, ha='center', fontweight='bold')

# FA2
ax1.plot(N, fa2_ms, 's--', color=FA2_COLOR, linewidth=2, markersize=8,
         label='Flash Attention 2 (full causal)', zorder=2)

# Sink FA (highlighted)
ax1.plot(N, sink_fa_ms, 'D-', color=SINK_COLOR, linewidth=3, markersize=9,
         label='Sink Flash Attention (ours)', zorder=3)
ax1.fill_between(N, [0]*len(N), sink_fa_ms, alpha=0.08, color=SINK_COLOR)

# Add OOM text for eager
ax1.annotate('OOM', xy=(16384, 25), fontsize=11, color=EAGER_COLOR,
             fontweight='bold', ha='center')
ax1.annotate('OOM', xy=(32768, 28), fontsize=11, color=EAGER_COLOR,
             fontweight='bold', ha='center')

# Crossover annotation
ax1.axvline(x=10000, color=SINK_COLOR, linestyle=':', alpha=0.5, linewidth=1.5)
ax1.annotate('crossover\nvs FA2', xy=(10000, 0.5), fontsize=9, color=SINK_COLOR,
             ha='center', style='italic', alpha=0.7)

ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.set_xlabel('Context Length (N)', fontsize=14)
ax1.set_ylabel('Forward Latency (ms)', fontsize=14)
ax1.set_title('Forward Pass Latency', fontsize=15, fontweight='bold', pad=12)
ax1.set_xticks(N)
ax1.set_xticklabels(['512', '1K', '2K', '4K', '8K', '16K', '32K'])
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.set_ylim(0.02, 40)

# Add speedup annotations
for i, n in enumerate(N):
    if fa2_ms[i] > 0 and sink_fa_ms[i] > 0 and n >= 16384:
        speedup = fa2_ms[i] / sink_fa_ms[i]
        ax1.annotate(f'{speedup:.1f}x', xy=(n, sink_fa_ms[i]),
                     xytext=(0, -18), textcoords='offset points',
                     fontsize=10, fontweight='bold', color=SINK_COLOR, ha='center')

# ============================================================
# Plot 2: Forward Pass Memory
# ============================================================
ax2 = axes[1]

# Eager memory
eager_valid_n_mem = [n for n, m in zip(N, eager_mem) if m is not None]
eager_valid_mem = [m for m in eager_mem if m is not None]
ax2.plot(eager_valid_n_mem, eager_valid_mem, 'o--', color=EAGER_COLOR, linewidth=2,
         markersize=8, label='Eager (materialized)', zorder=2)

# OOM markers for eager
ax2.annotate('OOM', xy=(16384, 12000), fontsize=11, color=EAGER_COLOR,
             fontweight='bold', ha='center')
ax2.annotate('OOM', xy=(32768, 15000), fontsize=11, color=EAGER_COLOR,
             fontweight='bold', ha='center')

# FA2 memory
ax2.plot(N, fa2_mem, 's--', color=FA2_COLOR, linewidth=2, markersize=8,
         label='Flash Attention 2 (full causal)', zorder=2)

# Sink FA memory (highlighted)
ax2.plot(N, sink_fa_mem, 'D-', color=SINK_COLOR, linewidth=3, markersize=9,
         label='Sink Flash Attention (ours)', zorder=3)
ax2.fill_between(N, [0]*len(N), sink_fa_mem, alpha=0.08, color=SINK_COLOR)

# Memory reduction annotation
ax2.annotate('128x less\nmemory', xy=(8192, 68), xytext=(8192, 2500),
             fontsize=11, fontweight='bold', color=SINK_COLOR,
             ha='center', arrowprops=dict(arrowstyle='->', color=SINK_COLOR, lw=1.5))

ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xlabel('Context Length (N)', fontsize=14)
ax2.set_ylabel('Peak Memory (MB)', fontsize=14)
ax2.set_title('Forward Pass Memory Usage', fontsize=15, fontweight='bold', pad=12)
ax2.set_xticks(N)
ax2.set_xticklabels(['512', '1K', '2K', '4K', '8K', '16K', '32K'])
ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax2.set_ylim(2, 20000)

plt.tight_layout(w_pad=3)
plt.savefig('/home/rulin/sink_attention/docs/performance.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Saved docs/performance.png")

# ============================================================
# Plot 3: Training (FWD+BWD) comparison
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6.5))

ax3 = axes2[0]
ax3.plot(N_train, fa2_train_ms, 's--', color=FA2_COLOR, linewidth=2, markersize=8,
         label='Flash Attention 2 (full causal)', zorder=2)
ax3.plot(N_train, sink_fa_train_ms, 'D-', color=SINK_COLOR, linewidth=3, markersize=9,
         label='Sink Flash Attention (ours)', zorder=3)
ax3.fill_between(N_train, [0]*len(N_train), sink_fa_train_ms, alpha=0.08, color=SINK_COLOR)

# Crossover
ax3.axvline(x=10000, color=SINK_COLOR, linestyle=':', alpha=0.5, linewidth=1.5)

# Speedup annotation at 16K
speedup_16k = fa2_train_ms[-1] / sink_fa_train_ms[-1]
ax3.annotate(f'{speedup_16k:.1f}x faster', xy=(16384, sink_fa_train_ms[-1]),
             xytext=(0, -20), textcoords='offset points',
             fontsize=11, fontweight='bold', color=SINK_COLOR, ha='center')

ax3.set_xscale('log', base=2)
ax3.set_yscale('log')
ax3.set_xlabel('Context Length (N)', fontsize=14)
ax3.set_ylabel('FWD + BWD Latency (ms)', fontsize=14)
ax3.set_title('Training Latency (Forward + Backward)', fontsize=15, fontweight='bold', pad=12)
ax3.set_xticks(N_train)
ax3.set_xticklabels(['512', '1K', '2K', '4K', '8K', '16K'])
ax3.legend(loc='upper left', fontsize=11, framealpha=0.9)

ax4 = axes2[1]
ax4.plot(N_train, fa2_train_mem, 's--', color=FA2_COLOR, linewidth=2, markersize=8,
         label='Flash Attention 2 (full causal)', zorder=2)
ax4.plot(N_train, sink_fa_train_mem, 'D-', color=SINK_COLOR, linewidth=3, markersize=9,
         label='Sink Flash Attention (ours)', zorder=3)
ax4.fill_between(N_train, [0]*len(N_train), sink_fa_train_mem, alpha=0.08, color=SINK_COLOR)

# Memory savings annotation
ax4.annotate('20% less memory', xy=(16384, sink_fa_train_mem[-1]),
             xytext=(0, -20), textcoords='offset points',
             fontsize=11, fontweight='bold', color=SINK_COLOR, ha='center')

ax4.set_xscale('log', base=2)
ax4.set_xlabel('Context Length (N)', fontsize=14)
ax4.set_ylabel('Peak Memory (MB)', fontsize=14)
ax4.set_title('Training Memory (Forward + Backward)', fontsize=15, fontweight='bold', pad=12)
ax4.set_xticks(N_train)
ax4.set_xticklabels(['512', '1K', '2K', '4K', '8K', '16K'])
ax4.legend(loc='upper left', fontsize=11, framealpha=0.9)

plt.tight_layout(w_pad=3)
plt.savefig('/home/rulin/sink_attention/docs/training.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

print("Saved docs/training.png")
