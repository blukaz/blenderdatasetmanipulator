import copy

import numpy as np
from numpy import linalg
import math
import matplotlib.pyplot as plt
import os
import OpenEXR
import Imath
from io import BytesIO
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

from scipy.spatial.transform import Rotation


EPOCHS = 12
BASE_PATH = "/home/bare/PycharmProjects/SyntheticDatasetGenerator/"


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

    channel = np.fromstring(channel_str, dtype=np.float32).reshape(size[1], size[0])*10

    return (channel)

def read_png(res):
    img = Image.open(BytesIO(res))
    return np.asarray(img)

def read_npy(res):
    return np.load(BytesIO(res))

def DepthConversion(PointDepth, fx, fy):
# https://github.com/unrealcv/unrealcv/issues/14#issuecomment-307028752
    H = PointDepth.shape[0]
    W = PointDepth.shape[1]
    i_c = (np.float(H) / 2) - 1
    j_c = (np.float(W) / 2) - 1
    columns, rows = np.meshgrid(np.linspace(0, W-1, num=W), np.linspace(0, H-1, num=H))
    DFC_rows = (i_c - rows)/fy
    DFC_cols = (j_c - columns)/fx
    #DistanceFromCenter = ((rows - i_c)**2 + (columns - j_c)**2)**(0.5)
    #PlaneDepth = PointDepth / ((1 + (DistanceFromCenter / f)**2)**(0.5))
    PlaneDepth = PointDepth / ((1 + (DFC_rows)**2 + (DFC_cols)**2)**(0.5))
    return PlaneDepth

def get_camera_param(depth):
    # camera intrinsics and depth conversion
    imageHeight, imageWidth = [480, 640]
    #print(imageWidth, imageHeight)
    CameraFOV = 90
    #fx = fy = f = imageWidth / (2 * math.tan(CameraFOV * (math.pi / 360)))
    fx = 355.55555555555554
    fy = 444.4444444444444
    #print(fx, fy)
    #depth = DepthConversion(depth, fx, fy)
    depth = depth.astype(np.float32)
    depth = o3d.geometry.Image(depth)
    return imageWidth, imageHeight, depth, fx, fy

def convert_rgbd2pcd(depth):
    imageWidth, imageHeight, img_depth, fx, fy = get_camera_param(depth)

    #rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img, img_depth, convert_rgb_to_intensity=False)
    #print(imageWidth, imageHeight)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=imageWidth, height=imageHeight, cx=319.5,
                                                    cy=239.5, fx=fx, fy=fy)
    # print(pinhole_cam.intrinsic_matrix)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(img_depth, intrinsic)
    #pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_cam)
    return pcd


list = os.listdir(BASE_PATH + "DepthMask/") # dir is your directory path
number_files = len(list)
print("Total number of data pairs: ", number_files)

for idx in range(number_files):
    print(idx)
    depth_path_idx = BASE_PATH + "DepthMask/kuka_depth_mask_" + "{:04d}".format(idx) + ".npy"
    #img_array = exr2numpy(depth_path_idx, channel_name='R')
    img_array = np.load(depth_path_idx, allow_pickle=True)
    #print(np.nanmax(img_array), np.nanmin(img_array))
    depth_max = np.nanmax(img_array)
    depth_min = np.nanmin(img_array)
    z_norm = (img_array - depth_min)/(depth_max - depth_min)
    #depth_o3d = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    #plt.imshow(z_norm, cmap="gray")
    #plt.show()
    #plt.imshow(img_array)
    #plt.show()

    # Pointcloud Generation from RGBD
    pcd = convert_rgbd2pcd(img_array)
    #pcd_tmp = copy.deepcopy(pcd)
    #print(pcd.get_max_bound(), pcd.get_min_bound())
    #o3d.visualization.draw_geometries([pcd])

    o3d.io.write_point_cloud( BASE_PATH + "KukaPointCloud/kuka_depth_mask_pcl_" + "{:04d}".format(idx) + ".ply", pcd)