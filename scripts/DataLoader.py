import numpy as np
import torch, pdb
import h5py
import torchvision
import random

from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries

class DataLoader(Dataset):
    
    def __init__(
        self, 
        file_path, 
        tile_width=320, 
        tile_height=240, 
        image_indices=None, 
        tiling=True, 
        augment=False,
        yolo=False,
        n_row_cells=None,
        n_col_cells=None):
        
        super().__init__()
        self.file = h5py.File(file_path, 'r')
        self.tile_width = tile_width
        self.tile_height = tile_width if tile_height is None else tile_height
        self.augment = augment
        self.resize = torchvision.transforms.Resize((tile_height, tile_width))
        self.image_indices = image_indices
        self.tiling = tiling
        self.yolo = yolo
        self.n_row_cells = n_row_cells
        self.n_col_cells = n_col_cells

        if self.yolo:
            assert self.n_col_cells is not None
            assert self.n_row_cells is not None

    def __getitem__(self, idx):

        if self.image_indices is not None:
            idx = self.image_indices[idx]
        
        # get image attributes
        image_width_min = int(self.file['attrs'][idx, 0])
        image_width_max = int(self.file['attrs'][idx, 1])
        image_height_min = int(self.file['attrs'][idx, 2])
        image_height_max = int(self.file['attrs'][idx, 3])
        mask_width_min = int(self.file['attrs'][idx, 4])
        mask_width_max = int(self.file['attrs'][idx, 5])
        mask_height_min = int(self.file['attrs'][idx, 6])
        mask_height_max = int(self.file['attrs'][idx, 7])
        
        if self.tiling:

            # generate tile dimensions
            height_to_width_ratio = self.tile_height / self.tile_width
            tile_width_min = mask_width_max-mask_width_min
            tile_width_max = np.min([
                (mask_height_max-mask_height_min)/height_to_width_ratio,
                mask_width_max,
                image_width_max-mask_width_min])
            tile_width = int(tile_width_min + np.random.random(1)[0]*(tile_width_max-tile_width_min))
            tile_height = int(height_to_width_ratio*tile_width)

            # generate tile location (bottom-left-corner)
            tile_x_min = mask_width_max - tile_width
            tile_x_max = mask_width_min
            tile_y_min = mask_height_min
            tile_y_max = mask_height_max - tile_height
            tile_x = int(tile_x_min + np.random.random(1)[0]*(tile_x_max-tile_x_min))
            tile_y = int(tile_y_min + np.random.random(1)[0]*(tile_y_max-tile_y_min))

            # extract tile
            tile = self.file['images'][idx, :, tile_y:tile_y+tile_height, tile_x:tile_x+tile_width]
            mask = self.file['masks'][idx, :, tile_y:tile_y+tile_height, tile_x:tile_x+tile_width]

        else:

            # use full image/mask
            tile = self.file['images'][idx]
            mask = self.file['masks'][idx]

        # convert to torch
        tile = torch.tensor(tile, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)

        # resize
        tile = self.resize(tile)
        mask = self.resize(mask)
        mask = torch.cat(
            [(mask>0.5).float(), (mask<0.5).float()],
            dim=0)

        if self.augment:

            # random flip
            if random.random() > 0.5:
                tile = torchvision.transforms.functional.hflip(tile)
                mask = torchvision.transforms.functional.hflip(mask)

            # random color augmentation
            tile = torchvision.transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                hue=0.1, 
                saturation=0.1)(tile)

        # convert mask to YOLO-type format
        if self.yolo:
            
            # initialize book keeping
            n_row_pixels = mask.shape[-2]
            n_col_pixels = mask.shape[-1]
            cell_row_pixels = int(n_row_pixels/self.n_row_cells)
            cell_col_pixels = int(n_col_pixels/self.n_col_cells)

            # find mask boundaries in pixel coords (numpy)
            boundary_mask = mask[0].detach().cpu().numpy()
            cell_boundary_locs = find_boundaries(boundary_mask)[cell_row_pixels//2::cell_row_pixels]
            cell_boundary_locs = np.argwhere(cell_boundary_locs).T
            cell_boundary_locs[0] = cell_boundary_locs[0]*cell_row_pixels + cell_row_pixels//2

            # YOLO-type output
            target = torch.zeros([2, self.n_row_cells, self.n_col_cells], dtype=torch.float)
            
            # fill in target using boundaries
            for row_pixel in np.unique(cell_boundary_locs[0]):
                
                # get left and right boundaries
                boundaries = np.sort(cell_boundary_locs[1][cell_boundary_locs[0]==row_pixel])
                col_pixel_l = boundaries[0]
                col_pixel_r = boundaries[-1]
                
                # convert to cell-coordinates
                cell_row = int(row_pixel/mask.shape[-2]*self.n_row_cells)
                
                cell_col_l = int(col_pixel_l/mask.shape[-1]*self.n_col_cells)
                cell_col_r = int(col_pixel_r/mask.shape[-1]*self.n_col_cells)
                cell_x_l = (col_pixel_l - (cell_col_l*cell_col_pixels)) / cell_col_pixels
                cell_x_r = (col_pixel_r - (cell_col_r*cell_col_pixels)) / cell_col_pixels
                
                # fill target information
                target[0, cell_row, cell_col_l] = 1.0
                target[0, cell_row, cell_col_r] = 1.0
                target[1, cell_row, cell_col_l] = cell_x_l
                target[1, cell_row, cell_col_r] = cell_x_r
                
            # overwrite mask
            mask = target
        
        return tile, mask

    def __len__(self):
        if self.image_indices is not None:
            N = len(self.image_indices)
        else:
            N = len(self.file['images'])
        return N
    
    def load_image(self, idx, width=None, height=None): # RGB: width=768, height=1024
        image = torch.tensor(self.file['images'][idx])
        if width is not None and height is not None:
            resize = torchvision.transforms.Resize((height, width))
            image = resize(image)
        return image
    
    def load_mask(self, idx, width=None, height=None):
        mask = torch.tensor(self.file['masks'][idx])
        if width is not None and height is not None:
            resize = torchvision.transforms.Resize((height, width))
            mask = resize(mask)
        mask = torch.round(mask)
        return mask

