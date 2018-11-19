from torch.utils.data.dataset import Dataset
import cv2
from glob import glob
import os
import PIL
from PIL import Image


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
        mask = PIL.Image.fromarray(cv2.imread(self.mask_list[index]))

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
