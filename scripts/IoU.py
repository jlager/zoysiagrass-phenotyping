import numpy as np

x_idx = 1
y_idx = 2
h_idx = 3
w_idx = 4
d_idx = 5
image_height = 64
image_width = 128
target_height = 4
target_width = 8

def IoU(pred, true):
    
    '''
    pred/true (array): shape [6, 4, 8] where grid cells are [4, 8]
                       and first axis corresponds to class probabilities,
                       x, y, h, w, and delay
    '''
    
    # find index location of max class probability
    pred_idx = np.unravel_index(np.argmax(pred[0], axis=None), pred[0].shape)
    true_idx = np.unravel_index(np.argmax(true[0], axis=None), true[0].shape)
    pred = pred[:,pred_idx[0],pred_idx[1]]
    true = true[:,true_idx[0],true_idx[1]]
    
    # if no pred object or no true object, then IoU=0
    if pred[0] < 0.5 or true[0] == 0:
        return 0.0

    #
    # pred mask
    #

    # convert x, y, h, w to pixel coords
    row = (pred[y_idx] + pred_idx[0])/target_height*image_height
    col = (pred[x_idx] + pred_idx[1])/target_width*image_width
    height = pred[h_idx]/target_height*image_height
    width = pred[w_idx]/target_width*image_width

    # determine the coordinates of the bounding box
    x_left = int(col - width)
    y_top = int(row + height)
    x_right = int(col + width)
    y_bottom = int(row - height)

    # build pred segmentation mask
    pred_mask = np.ones([image_height, image_width])
    pred_mask[:y_bottom] = 0
    pred_mask[y_top:] = 0
    pred_mask[:, :x_left] = 0
    pred_mask[:, x_right:] = 0

    #
    # true mask
    #

    # convert x, y, h, w to pixel coords
    row = (true[y_idx] + true_idx[0])/target_height*image_height
    col = (true[x_idx] + true_idx[1])/target_width*image_width
    height = true[h_idx]/target_height*image_height
    width = true[w_idx]/target_width*image_width

    # determine the coordinates of the bounding box
    x_left = int(col - width)
    y_top = int(row + height)
    x_right = int(col + width)
    y_bottom = int(row - height)

    # build true segmentation mask
    true_mask = np.ones([image_height, image_width])
    true_mask[:y_bottom] = 0
    true_mask[y_top:] = 0
    true_mask[:, :x_left] = 0
    true_mask[:, x_right:] = 0

    # compute IoU
    intersection = np.bitwise_and(pred_mask.astype(bool), true_mask.astype(bool)).sum()
    union = np.bitwise_or(pred_mask.astype(bool), true_mask.astype(bool)).sum()
    return intersection/union