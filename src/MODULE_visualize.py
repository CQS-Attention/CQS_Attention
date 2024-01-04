import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random

def subsequence_length_hist(subsequence_lenghts):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.hist(subsequence_lenghts)
    ax.locator_params(axis='y', integer=True)
    ax.locator_params(axis='x', integer=True)
    ax.set_xlabel('subsequence length')
    ax.set_ylabel('Count')    
    plt.show()

def P_partition(N, W, TL, show_grid = False, save_path = None):
    W_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(W)]
    grid_color = 'black' if show_grid else None
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim([0,10])
    ax.set_ylim([0, 10])
    ax.set_xticks([])    
    ax.set_yticks([])
    cell_size = 10 / N

    for i in range(W):
        # all cells for current worker are listed in TL[i]
        for x,y in TL[i]:
            rec = Rectangle((x*cell_size, (N-1-y)*cell_size), cell_size, cell_size, facecolor = W_colors[i], alpha = 0.5, edgecolor = grid_color)
            fig.gca().add_patch(rec)
    ax.set_aspect('equal', adjustable='box')
    if save_path != None:
        file_name = f'{save_path}/N{N}_W{W}_P_partition.png'
        plt.savefig(file_name)
    plt.show()
