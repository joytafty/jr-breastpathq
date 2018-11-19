# Representation Learning for Nuclear Segmentation
# Use external data: /Users/trimchala/BreastPathQ/breastpathq/datasets/MoNuSegTraining/Annotations/
# https://monuseg.grand-challenge.org/Data/
# Predict on provided data in cells folder
import skimage
import numpy as np
import tqdm
from torch.autograd import Variable
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def chw_to_hwc(img):
    return np.transpose(img, axes=[1, 2, 0])

def visualize_learning(model, dataloader):
    for data in tqdm(dataloader):
        data, target = Variable(data["image"], volatile=True), Variable(data["mask"])
        output = model(data.float())

    for input_instance, target_instance, output_instance in zip(data, target, output):
        input_hwc = chw_to_hwc(input_instance.detach().numpy())
        target_hwc =  skimage.color.rgb2gray(chw_to_hwc(target_instance.detach().numpy()))
        output_hwc = skimage.color.rgb2gray(chw_to_hwc(output_instance.detach().numpy()))

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        axes[0].imshow(input_hwc)
        axes[1].imshow(target_hwc, cmap='gray')
        axes[2].imshow(output_hwc, cmap='gray')