from utils import *
import os
from glob import glob
import pandas as pd
import numpy as np
import PIL
from PIL import Image

project_path = os.path.normpath(os.path.join(os.getcwd(), '../'))
train_path = os.path.join(project_path, 'datasets/train/')
cells_path = os.path.join(project_path, 'datasets/cells/')
train_tifs = glob(train_path + "*.tif")
cells_tifs = glob(cells_path + "*.tif")
cells_xmls = glob(cells_path + "*.xml")
train_labels = pd.read_csv(os.path.join(project_path, 'datasets/train_labels.csv'))

# Data Set Loader
