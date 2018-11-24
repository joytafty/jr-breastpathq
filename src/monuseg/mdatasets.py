from torch.utils.data.dataset import Dataset
import cv2
from glob import glob
import os
import PIL
from PIL import Image
import numpy as np


class NuclearMaskedDataSet(Dataset):
    def __init__(self, image_path, mask_path, transforms=None):
        # stuff
        self.image_path = image_path
        self.mask_path = mask_path
        self.transforms = transforms
        self.image_list = glob(os.path.join(self.image_path, "*.tif"))
        self.mask_list = glob(os.path.join(self.mask_path, "*.png"))

    def __getitem__(self, index):
        img = PIL.Image.fromarray(cv2.imread(self.image_list[index]))
        mask_gray = PIL.Image.fromarray(cv2.imread(self.mask_list[index])).convert('L')

        # This follows the unet convention
        # We should put an option arg to specify which network to train with
        ch1 = (mask_gray == mask_gray.min()).astype(int)
        ch2 = (mask_gray > mask_gray.min()).astype(int)
        mask = np.dstack([ch1, ch2])

        # Transform only on training set
        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        return {
            "image": img,
            "mask": mask
        }

    def __len__(self):
        return len(self.image_list)
