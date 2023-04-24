import camera
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import argparse

"""
This script helps to visualize the camera position in the global map
1. pose2xyz: It is used to get 3D point from homohenous matrix
2. visualize_camera: Visualize Single camera in both 2D(x, y) and 3D(x, y, z) format
3. visualize_multi_camera: Visualize Multiple camera in both 2D(x, y) and 3D(x, y, z) format to compare relative position
"""
def pose2xyz(camera, shift=False):
    xyz_list = []
    if shift:
        for idx in range(len(camera)):
            xyz = np.subtract(camera[idx][1][:3, 3], np.asarray([[332400], [8375600], [0]]))
            xyz = [float(x) for x in xyz]
            xyz_list.append(xyz)
    else:
        for idx in range(len(camera)):
            xyz = camera[idx][1][:3, 3]
            xyz = [float(x) for x in xyz]
            xyz_list.append(xyz)
    xyz_np = np.asarray(xyz_list, dtype="object")
    return xyz_np


def visualize_camera(xyz, mode="3d", filename=0):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    # Create the figure and the 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Create the scatter plot
    # save file name in current time format if not specified in a folder name Result
    if mode == "2d":
        if filename == 0:
            now = datetime.now()
            filename = "result/" + now.strftime("%Y-%m-%d %H:%M:%S") + "_2d.png"
        else:
            filename = "result/" + filename + "_2d.png"
        scatter = ax.scatter(x, y, np.zeros(x.shape), c=z, cmap='viridis', s=.5, alpha=0.9)
        fig.colorbar(scatter, pad = 0.15, label = "Camera depth")
    else:
        if filename == 0:
            now = datetime.now()
            filename = "result/" + now.strftime("%Y-%m-%d %H:%M:%S") + "_3d.png"
        else:
            filename = "result/" + filename + "_3d.png"
        scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=.5, alpha=0.9)
        fig.colorbar(scatter, pad = 0.15, label = "Camera depth")

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Save the figure in high resolution
    if os.path.exists(os.getcwd() + "/result"):
        plt.savefig(filename, dpi=600)
    else:
        os.mkdir("result")
        plt.savefig(filename, dpi=600)
    # Show the plot
    plt.show()


def visualize_multi_camera(xyzs, label_name, mode="3d", filename=0):
    # Create the figure and the 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # Create the scatter plot
    if mode == "2d":
        if filename == 0:
            now = datetime.now()
            filename = "result/" + now.strftime("%Y-%m-%d %H:%M:%S") + "multi_2d.png"
        else:
            filename = "result/" + filename + "_multi2d.png"
        for idx in range(len(xyzs)):
            xyz = xyzs[idx]
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]
            ax.scatter(x, y, np.zeros(x.shape), s=.5, alpha=0.9, label=label_name[idx])
    else:
        if filename == 0:
            now = datetime.now()
            filename = "result/" + now.strftime("%Y-%m-%d %H:%M:%S") + "_multi_3d.png"
        else:
            filename = "result/" + filename + "_multi_3d.png"
        for idx in range(len(xyzs)):
            xyz = xyzs[idx]
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]
            ax.scatter(x, y, z, s=.5, alpha=0.9, label = label_name[idx])

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    # Save the figure in high resolution
    if os.path.exists(os.getcwd() + "/result"):
        plt.savefig(filename, dpi=1200)
    else:
        os.mkdir("result")
        plt.savefig(filename, dpi=1200)
    # Show the plot
    plt.show()


def main(camera_xml, opencv_xml, labels, shift = False):
    camera_xyz = []
    for idx in range(len(camera_xml)):
        camera_set = camera.main(camera_xml[idx], opencv_xml[idx])
        xyz_1 = pose2xyz(camera_set, shift)
        camera_xyz.append(xyz_1)
        visualize_camera(xyz_1, "2d", filename = labels[idx])
        visualize_camera(xyz_1, "3d", filename = labels[idx])
    if len(xyz_1)>1:
        visualize_multi_camera(camera_xyz, labels, "2d", "camera")
        visualize_multi_camera(camera_xyz, labels, "3d", "camera")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_xml", default=["/home/turin/Documents/GitHub/long_term_underwater_vision/dataset_file/2014/cam14.xml"], type=list, help="Camera XML list or array")
    parser.add_argument("--opencv_xml", default=["/home/turin/Documents/GitHub/long_term_underwater_vision/dataset_file/opencv_cam_calib.xml"], type=list, help="Camera XML list or array")
    cam14_xml = "/home/turin/Documents/GitHub/long_term_underwater_vision/dataset_file/2014/cam14.xml"
    cam15_xml = "/home/turin/Documents/GitHub/long_term_underwater_vision/dataset_file/2015/cam15.xml"
    opencv_xml = "/home/turin/Documents/GitHub/long_term_underwater_vision/dataset_file/opencv_cam_calib.xml"
    camera14 = camera.main(cam14_xml, opencv_xml)
    camera15 = camera.main(cam15_xml, opencv_xml)
    xyz14 = pose2xyz(camera14, True)
    xyz15 = pose2xyz(camera15, True)
    visualize_camera(xyz14, "3d", "cam14")
    visualize_camera(xyz15, "3d", "cam15")
    visualize_camera(xyz14, "2d", "cam14")
    visualize_camera(xyz15, "2d", "cam15")
    visualize_multi_camera([xyz14, xyz15], ["cam14", "cam15"], "2d", "multi")
    visualize_multi_camera([xyz14, xyz15], ["cam14", "cam15"], "3d", "multi")
