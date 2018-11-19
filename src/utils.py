import PIL
from PIL import Image
import os
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import tqdm


def get_meta(path, file_type, glob_col = "file_path"):
    file_list = glob(os.path.join(path, f"*.{file_type}"))
    meta_df = pd.DataFrame(file_list, columns=[glob_col])
    meta_df["file_name"] = meta_df[glob_col].apply(lambda f: os.path.basename(f))
    meta_df["item_name"] = meta_df["file_name"].apply(lambda g: os.path.splitext(g)[0])

    return meta_df


# Utility code for parsing cell xml
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


# Parse session xml
def get_cells_sessions_dfs(cells_session_path):
    # cells_session_path = os.path.join(cells_path, "Sedeen", "*")
    all_sessions_regions_pts_dfs = pd.DataFrame()

    for session_file in tqdm(glob(cells_session_path)):
        with open(session_file, 'r') as f:
            raw_xml = f.readlines()
            graphic_list = get_ptslist_from_str(raw_xml, graphic_tag="graphic")

        pts_dfs = pd.DataFrame()
        for graphic_elem in graphic_list:
            pts_df = get_pts_from_graphic(graphic_elem)
            pts_dfs = pd.concat([pts_dfs, pts_df], axis=0)

        pts_dfs['session_file'] = session_file
        all_sessions_regions_pts_dfs = pd.concat([all_sessions_regions_pts_dfs, pts_dfs], axis=0)

    all_sessions_regions_pts_dfs['base_name'] = all_sessions_regions_pts_dfs['session_file'].apply(
        lambda x: os.path.basename(x).replace('_crop.session.xml', '')
    )

    return all_sessions_regions_pts_dfs

