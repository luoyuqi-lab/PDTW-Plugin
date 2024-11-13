import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from Functions import cdtw, pdtw, fill_envelope


config = {
    "font.family": 'sans serif',
    "font.serif": ['Arial', ]
}
plt.rcParams.update(config)

def assemble_extra_data(a, b, N, r):
    """
    :param a: Time series a
    :param b: Time series b
    :param r: warping window size in CDTW
    :param N: Divide time series into n parts.
    :return: pairwise subsequences of n (number of slicing) time series partitions.
    """
    x, y = [], []
    x1, y1 = [], []
    if len(a) == len(b):
        l = math.floor(len(a) / N)
        x.append(a[0:l])
        y.append(b[0:l])
        x1.append([0] + x[0] + a[l:l + r + 1])
        y1.append([0] + y[0] + b[l:l + r + 1])
        if N != 2:
            for i in range(1, N - 1):
                p = a[int(i * l): int((i + 1) * l)]
                x.append(p)
                p = b[int(0 + i * l): int((i + 1) * l)]
                y.append(p)
                x1.append([1] + a[max(0, i * l - (r + 1)):min(len(a)-1,(i + 1) * l + r + 1)])
                y1.append([1] + b[max(0, i * l - (r + 1)):min(len(a)-1,(i + 1) * l + r + 1)])
                i += 1
        x.append(a[(N - 1) * l:])
        y.append(b[(N - 1) * l:])
        x1.append([2] + a[(N - 1) * l - (r + 1): (N - 1) * l] + x[N - 1])
        y1.append([2] + b[(N - 1) * l - (r + 1): (N - 1) * l] + y[N - 1])
    else:
        print('Please align! Now we do not support different time stamps.')
    return x1, y1

# Example time series
a = [11,14,20,23,25,28,29,29,29,28,27,25,22,19,15,11, 8, 6,4,1,0,0,1,3,7,10,13,15,19,21,22,23,23,22,21,19,17,13,10, 5]
b = [8,12,16,18,20,22,23,24,25,25,24,23,22,20,19,16,13,12,9,7,6,5,4,4,5, 6, 7, 8,10,12,13,14,15,16,17,18,18,17,16,14]
r = 3
U, L = fill_envelope(b, r)
colors = ['#BD4146', '#276C9E','#1E803D','#F37020', '#cfe7c4', '#AED4E5']

c, d = assemble_extra_data(a, b, 5, r)
path1 = pdtw(c[0], d[0], r)[1]
path1 = [(i , j ) for (i, j) in path1]
path2 = pdtw(c[3], d[3], r)[1]
path2 = [(i + 20, j + 20) for (i, j) in path2]

# Time series and envelope
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(a) + 1), a, label='Series T', color=colors[0])
ax.plot(range(1, len(b) + 1), b, label='Series S (with envelope)', color=colors[1])

# Get the index range of path1 and path2
path1_indices = set(i for i, j in path1)
path2_indices = set(i for i, j in path2)
all_path_indices = path1_indices.union(path2_indices)

# background
ax.axvspan(1, 9, color=colors[5], alpha=0.5, ymin=0, ymax=30/30)
ax.axvspan(9, 25, color=colors[4], alpha=0.5, ymin=0, ymax=30/30)
ax.axvspan(25, 33, color=colors[5], alpha=0.5, ymin=0, ymax=30/30)
ax.axvspan(33, 40, color=colors[4], alpha=0.5, ymin=0, ymax=30/30)

# Envelope and LB_Keogh distance
lb_keogh_label_added = False
for i in range(len(a)):
    if i not in all_path_indices:
        if a[i] > U[i]:
            if not lb_keogh_label_added:
                ax.plot([i + 1, i + 1], [U[i], a[i]], color=colors[2], linestyle='--', linewidth=1, label='LB_Keogh distance')
                lb_keogh_label_added = True
            else:
                ax.plot([i + 1, i + 1], [U[i], a[i]], color=colors[2], linestyle='--', linewidth=1)
        elif a[i] < L[i]:
            ax.plot([i + 1, i + 1], [L[i], a[i]], color=colors[2], linestyle='--', linewidth=1)

# DTW path
alignment_label_added = False
for (i, j) in path1 + path2:
    if not alignment_label_added:
        ax.plot([i + 1, j + 1], [a[i], b[j]], color=colors[3], label='PDTW alignment', linestyle='--', linewidth=1)
        alignment_label_added = True
    else:
        ax.plot([i + 1, j + 1], [a[i], b[j]], color=colors[3], linestyle='--', linewidth=1)

import numpy as np

# draw envelope 和 envelope region, exclude path1 and path2 regions
start = 0
for i in range(len(a)):
    if i in all_path_indices:
        if start < i:
            x_values = np.arange(start + 1, i + 2)
            ax.plot(x_values, U[start:i+1], linestyle='--', color='#9284B4')
            ax.plot(x_values, L[start:i+1], linestyle='--', color='#C0A3C0')
            ax.fill_between(x_values, U[start:i+1], L[start:i+1], color='#DFD6D6', alpha=0.2)
        start = i + 1

if start < len(a):
    x_values = np.arange(start + 1, len(U) + 1)
    ax.plot(x_values, U[start:], label='Partial upper envelope', linestyle='--', color='#9284B4')
    ax.plot(x_values, L[start:], label='Partial lower envelope', linestyle='--', color='#C0A3C0')
    ax.fill_between(x_values, U[start:], L[start:], color='#DFD6D6', alpha=0.2, label='Envelope region')

ax.legend(loc=(0.045,0.001), fontsize=9.5)

legend_elements = [
    patches.Patch(facecolor=colors[4], edgecolor='none', label='LB_Keogh region'),
    patches.Patch(facecolor=colors[5], edgecolor='none', label='PDTW region')
]
ax2 = fig.add_axes([0.15, 0.75, 0.1, 0.1], frameon=False)
ax2.legend(handles=legend_elements, fontsize=9.5,loc=(5.55,0.58))
ax2.axis('off')

ax.set_xlabel('Time', fontsize=12, loc='right')
ax.set_ylabel('Value', fontsize=12, loc='top')
ax.set_xticks([1, 10, 20, 30, 40])
ax.set_yticks([0, 10, 20, 30])
ax.tick_params(axis='both', which='major', labelsize=12)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.xaxis.set_label_coords(0.97, -0.08)

plt.show()