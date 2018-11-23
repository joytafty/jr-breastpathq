# Representation Learning for Nuclear Segmentation
# Use external data: /Users/trimchala/BreastPathQ/breastpathq/datasets/MoNuSegTraining/Annotations/
# https://monuseg.grand-challenge.org/Data/
# Predict on provided data in cells folder
import skimage
import numpy as np
import pandas as pd
import tqdm
from torch.autograd import Variable
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
from skimage import data, color
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

# Create cell mask from cells_tif
# First read in cell xml
def get_ptslist_from_str(xml_str, graphic_tag='graphic'):
    elem_open_indices = [idx for (idx, line) in enumerate(xml_str) if f"<{graphic_tag}" in line]
    elem_close_indices = [idx for (idx, line) in enumerate(xml_str) if f"</{graphic_tag}" in line]

    elem_lines = [
        [l.strip() for l in xml_str[start:end+1]]
        for (start, end) in zip(elem_open_indices, elem_close_indices)
    ]

    return [ET.XML('\n'.join(elem_line)) for elem_line in elem_lines]

def get_pts_from_graphic(graphic_elem, point_list_tag="point-list"):

    point_list_elem = graphic_elem.findall("point-list")[0]

    pts_df = pd.DataFrame([
        [int(float(p)) for p in point_elem.text.split(",")]
        for point_elem in point_list_elem
    ], columns=["x","y"])

    for key, val in graphic_elem.attrib.items():
        pts_df[key] = [val]*len(pts_df)

    return pts_df


def get_pts_from_regions(region_elem, vertex_tag="Vertex"):
    vertices_elem = region_elem.findall("Vertices")[0].findall(vertex_tag)
    pts_df = pd.DataFrame([vertex_elem.attrib for vertex_elem in vertices_elem])
    pts_df.columns=['x', 'y']

    for key, val in region_elem.attrib.items():
        pts_df[key] = [val]*len(pts_df)

    return pts_df


def get_nuc_regions(raw_nuc_xml, graphic_tag='Regions', elem_tag="Region"):
    regions_elem = get_ptslist_from_str(xml_str=raw_nuc_xml, graphic_tag=graphic_tag)
    region_elem_list = regions_elem[0].findall(elem_tag)
    region_elem = region_elem_list[0]

    region_pts_dfs = pd.DataFrame()
    for region_elem in region_elem_list:
        region_pts_df = get_pts_from_regions(region_elem)
        region_pts_dfs = pd.concat([region_pts_dfs, region_pts_df], axis=0)

    region_pts_dfs['x'] = region_pts_dfs['x'].astype(float)
    region_pts_dfs['y'] = region_pts_dfs['y'].astype(float)

    return region_pts_dfs


def get_bounded_pts(img, region_pts_dfs):
    x, y = np.meshgrid(np.arange(img.size[0]), np.arange(img.size[1])) # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T

    num_regions = len(region_pts_dfs['Id'].unique())
    nuc_mask = np.zeros((img.size[0], img.size[1], num_regions), dtype=float)

    for idx, nuc_id in tqdm(enumerate(region_pts_dfs['Id'].unique()), total=num_regions):
        grp = region_pts_dfs[region_pts_dfs['Id']==nuc_id]
        p = Path(list(zip(grp['x'], grp['y'])))
        grid = p.contains_points(points)
        mask = grid.reshape(img.size[0], img.size[1]).astype(int)
        nuc_mask[:, :, idx] = mask*idx

    nuc_groundtruth = np.amax(nuc_mask, axis=2).astype(int)

    return nuc_groundtruth


def crop_image_and_mask(image, mask, centroid_df, num_grid = 4, item_name="unknown", display_crop=False):
    image_w, image_h, image_c = image.shape
    mask_w, mask_h, _ = mask.shape

    x = np.linspace(0, image_w, num_grid).astype(int)
    y = np.linspace(0, image_h, num_grid).astype(int)

    image_crops = []
    mask_crops = []
    centroid_crops = []
    for x1, x2 in zip(x[:-1], x[1:]):
        for y1, y2 in zip(y[:-1], y[1:]):
            image_crops.append(image[x1:x2, y1:y2, :])
            mask_crops.append(mask[x1:x2, y1:y2])

            adjusted_centroid_df = centroid_df.apply(
                lambda row:
                (x1 < row['x'] <= x2) and (y1 < row['y'] <= y2),
                axis=1
            ).reset_index(drop=True)

            adjusted_centroid_df = centroid_df[
                centroid_df.apply(
                    lambda row: (x1 < row['x'] <= x2) and (y1 < row['y'] <= y2), axis=1)
            ].reset_index(drop=True)
            adjusted_centroid_df['x'] = adjusted_centroid_df['x'] - x1
            adjusted_centroid_df['y'] = adjusted_centroid_df['y'] - y1

            centroid_crops.append(adjusted_centroid_df)

    if display_crop:
        sidx = np.random.randint(low=0, high=len(image_crops), size=1)[0]
        image_hsv = color.rgb2hsv(image_crops[sidx])
        mask_hsv = mask_crops[sidx]
        mask_hsv = (mask_crops[sidx][:, :, :3] > np.min(mask_crops[sidx][:, :, :3])).astype(int)

        alpha=0.5
        image_hsv[..., 0] = mask_hsv[..., 0] * (1-alpha)
        image_hsv[..., 1] = mask_hsv[..., 1] * alpha

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_crops[sidx])
        axes[1].imshow(mask_crops[sidx])
        axes[2].imshow(color.hsv2rgb(image_hsv))

        plt.suptitle(f'cropped_{item_name}_subregion_{sidx}')

        plt.show()

    return image_crops, mask_crops, centroid_crops