import torch
import numpy as np

def convert(tile, target):

    '''
    tile: [h, w, c]
    target: [2, n_row_cells, n_col_cells]
    '''

    n_row_cells = target.shape[-2]
    n_col_cells = target.shape[-1]
    cell_row_pixels = int(tile.shape[0]/n_row_cells)
    cell_col_pixels = int(tile.shape[1]/n_col_cells)

    # find boundary points
    plot_mask = np.zeros_like(tile[:,:,0])
    locs = np.argwhere(target[0]==1)
    for loc in locs:
        row = int(loc[0]*cell_row_pixels + cell_row_pixels//2)
        col = int(loc[1]*cell_col_pixels + target[1,loc[0],loc[1]]*cell_col_pixels)
        for i in range(row-3, row+4):
            for j in range(col-3, col+4):
                if (i >= 0) and (i < plot_mask.shape[-2]) and (j >= 0) and (j < plot_mask.shape[-1]):
                    plot_mask[i, j] = 1
                    tile[i,j] = 0
    
    mask_filter = np.concatenate([
        plot_mask[:,:,None],
        np.zeros_like(plot_mask[:,:,None]),
        np.zeros_like(plot_mask[:,:,None])], axis=-1)

    return mask_filter