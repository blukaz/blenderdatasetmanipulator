import OpenEXR
import Imath
from PIL import Image
import array
import numpy as np
import json

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['figure.figsize'] = [20, 20]

BASE_PATH = "/home/bare/PycharmProjects/SyntheticDatasetGenerator/TestData/"

def exr2numpy(exr_path, channel_name):
    '''
    See:
    https://excamera.com/articles/26/doc/intro.html
    http://www.tobias-weis.de/groundtruth-data-for-computer-vision-with-blender/
    '''
    file = OpenEXR.InputFile(exr_path)
    dw = file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    Float_Type = Imath.PixelType(Imath.PixelType.FLOAT)

    channel_str = file.channel(channel_name, Float_Type)

    channel = np.fromstring(channel_str, dtype=np.float32).reshape(size[1], size[0])

    return (channel)

list = os.listdir(BASE_PATH + "SegmentationMask/") # dir is your directory path
number_files = len(list)
print("Total number of data pairs: ", number_files)


print("Converting...")

for idx in range(number_files):
    print(idx)
    exr_path = BASE_PATH + "SegmentationMask/" + "kuka_mask_" + "{:04d}".format(idx) + ".exr"
    semantic_index = exr2numpy(exr_path, channel_name='R')
    #fig = plt.figure()
    #plt.imshow(semantic_index)
    #plt.colorbar()
    #plt.show()

    exr_path = BASE_PATH + "Depth/" + "kuka_depth_" + "{:04d}".format(idx) + ".exr"
    depth = exr2numpy(exr_path, channel_name='R')
    depth *=10
    #fig = plt.figure()
    #print(np.amin(depth), np.amax(depth))
    #plt.imshow(depth)
    #plt.colorbar()
    #plt.show()

    #print(depth.shape, semantic_index.shape)

    depth_mask = np.multiply(depth, semantic_index)
    #fig = plt.figure()
    #plt.imshow(depth_mask)
    #plt.colorbar()
    #plt.show()
    np.save(BASE_PATH + "DepthMask/kuka_depth_mask_" + "{:04d}".format(idx) + ".npy", depth_mask)

print("Finished!")