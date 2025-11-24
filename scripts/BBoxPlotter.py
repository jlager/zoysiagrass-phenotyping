import numpy as np
import pdb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot(ax, image, pred=None, true=None):

    '''
    ax: current axis handle
    image: np array with shape [H, W, C]
    pred: predicted target array with shape [T, 4, 8]
          T: class + x + y + h + w + delay = 6
    true: ground truth target array with [class, x, y, h, w, delay]
    '''

    x_idx = 1
    y_idx = 2
    h_idx = 3
    w_idx = 4
    d_idx = 5
    image_height = 64
    image_width = 128
    target_height = 4
    target_width = 8

    if true is not None:

        idx = np.unravel_index(np.argmax(true[0], axis=None), true[0].shape)
        cell_row, cell_col = idx[0], idx[1]
        
        if true[0, cell_row, cell_col] >= 0.5:
            
            # extract data
            row = true[y_idx, cell_row, cell_col]
            col = true[x_idx, cell_row, cell_col]
            height = true[h_idx, cell_row, cell_col]
            width = true[w_idx, cell_row, cell_col]

            # convert to pixels
            row = (row + cell_row)/target_height*image_height
            col = (col + cell_col)/target_width*image_width
            height = height/target_height*image_height
            width = width/target_width*image_width
            
            # construct bounding box
            x = col - width
            y = row - height
            ax.add_patch(Rectangle(
                [x,y],
                2*width,
                2*height,
                linewidth=2,
                edgecolor='green',
                facecolor='none'))

            # include delay
            delay = np.round(true[d_idx, cell_row, cell_col]*12, 1)
            ax.text(
                x=x/image.shape[1]-0.025,
                y=1-y/image.shape[0]-0.05,
                s='d = {0}'.format(delay),
                c='white', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='green'),
                fontsize='x-small',
                ha='right')

    if pred is not None:

        idx = np.unravel_index(np.argmax(pred[0], axis=None), pred[0].shape)
        cell_row, cell_col = idx[0], idx[1]
        
        if pred[0, cell_row, cell_col] >= 0.5:

            # extract data
            row = pred[y_idx, cell_row, cell_col]
            col = pred[x_idx, cell_row, cell_col]
            height = pred[h_idx, cell_row, cell_col]
            width = pred[w_idx, cell_row, cell_col]

            # convert to pixels
            row = (row + cell_row)/target_height*image_height
            col = (col + cell_col)/target_width*image_width
            height = height/target_height*image_height
            width = width/target_width*image_width
            
            # construct bounding box
            x = col - width
            y = row - height
            ax.add_patch(Rectangle(
                [x,y],
                2*width,
                2*height,
                linewidth=2,
                edgecolor='red',
                facecolor='none'))

            # include delay
            delay = np.round(pred[d_idx, cell_row, cell_col]*12, 1)   
            ax.text(
                x=(x+2*width)/image.shape[1]+0.04,
                y=1-y/image.shape[0]-0.05,
                s='d = {0}'.format(delay),
                c='white', 
                transform=ax.transAxes, 
                bbox=dict(facecolor='red'),
                fontsize='x-small')

    return ax
        