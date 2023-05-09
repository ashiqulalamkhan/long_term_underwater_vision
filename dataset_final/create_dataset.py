import numpy as np
import vis_camera
import camera
import os
import pickle
import shutil
import preprocess_dataset

##GET TRANSFORMED DATASET:
"""
1. Point Cloud Registration [source: pcd15, pcd16, target: pcd14]
2. Camera Pose Transformation  [camera15, camera16 transformed to camera14 reference frame]
3. Return [camera id, transformed_camera_pose]
"""
transformed_camera_sets = preprocess_dataset.camera_transformation()
camera_14, camera_15, camera_16, cam14cls, cam15cls, cam16cls = transformed_camera_sets
camera_14 = np.asarray(camera_14, dtype="object")
camera_15 = np.asarray(camera_15, dtype="object")
camera_16 = np.asarray(camera_16, dtype="object")
cam14cls = np.asarray(cam14cls, dtype="object")
cam15cls = np.asarray(cam15cls, dtype="object")
cam16cls = np.asarray(cam16cls, dtype="object")




###CLASS FOR POSITIVE DATASET
class CreatePositiveDataSet:
    def __init__(self, camera_base, camera_base_path, camera_two, camera_two_path, max_depth, thres = 0.25):
        self.camera_base = camera_base
        self.camera_base_path = camera_base_path
        self.camera_two = camera_two
        self.camera_two_path = camera_two_path
        self.max_depth = max_depth
        self.thres = thres


    def create_positive_sample(self):
        max_depth = self.max_depth
        positive_sample = []
        z = []
        ratio_all = []
        for idx in range(len(self.camera_base)):
            for idy in range(len(self.camera_two)):
                x_diff = self.camera_base[idx][1][0, 3]-self.camera_two[idy][1][0, 3]
                y_diff = self.camera_base[idx][1][1, 3]-self.camera_two[idy][1][1, 3]
                z_diff = self.camera_base[idx][1][2, 3]-self.camera_two[idy][1][2, 3]
                if z_diff<0:
                    ratio = -(.10*(max_depth-z_diff)/max_depth)
                else:
                    ratio = (.10*(max_depth-z_diff)/max_depth)
                if abs(x_diff)<(self.thres - ratio) and abs(y_diff)<(self.thres - ratio):
                    z.append(z_diff)
                    ratio_all.append(ratio)
                    positive_sample.append([self.camera_base[idx], self.camera_two[idy]])
        return  positive_sample

    def save_train_dataset(self):
        positive_sample = self.create_positive_sample()
        print(len(positive_sample))
        if not os.path.exists(os.getcwd() + "/train_positive_pair"):
            os.mkdir(os.getcwd() + "/train_positive_pair")
        for idx in positive_sample:
            dst = os.getcwd() + "/train_positive_pair/" + idx[0][0][:-4]
            if not os.path.exists(dst):
                os.mkdir(dst)
                shutil.copy(self.camera_base_path + idx[0][0], dst)
            src = self.camera_two_path + idx[1][0]
            shutil.copy(src, dst)
        return positive_sample

    def save_test_dataset(self):
        positive_sample = self.create_positive_sample()
        if not os.path.exists(os.getcwd() + "/test_positive_pair"):
            os.mkdir(os.getcwd() + "/test_positive_pair")
        for idx in positive_sample:
            dst = os.getcwd() + "/test_positive_pair/" + idx[0][0][:-4]
            if not os.path.exists(dst):
                os.mkdir(dst)
                shutil.copy(self.camera_base_path + idx[0][0], dst)
            src = self.camera_two_path + idx[1][0]
            shutil.copy(src, dst)
        return positive_sample

##GET MIN MAX DEPTH
def depth_min_max(camera_base, camera_two):
    z = []
    for idx in range(len(camera_base)):
        for idy in range(len(camera_two)):
            z_diff = camera_base[idx][1][2, 3]-camera_two[idy][1][2, 3]
            z.append(z_diff)
    return min(z), max(z)



#File Path
path_14 = "/home/turin/Desktop/lizard_island/jackson/chronological/2014/r20141102_074952_lizard_d2_081_horseshoe_circle01/081_photos/"
path_15 = "/home/turin/Desktop/lizard_island/jackson/chronological/2015/r20151207_222558_lizard_d2_039_horseshoe_circle01/039_photos/"
path_16 = "/home/turin/Desktop/lizard_island/jackson/chronological/2016/r20161121_063027_lizard_d2_050_horseshoe_circle01/050_photos/"
#Get min max depth ratio compared to base dataset
min_z_train, max_z_train = depth_min_max(camera_14, camera_15)
min_z_test, max_z_test = depth_min_max(camera_14, camera_16)
#Get a Ratio of max min depth to create threshold for each year dataset
ratio = (max_z_test - min_z_test)/(max_z_train - min_z_train)
print(ratio)
#Multiply the ratio with threshold of train dataset
thres_train, thres_test = .2*(ratio+.1), .2
#Create test and train positive dataset
positive_sample_train = CreatePositiveDataSet(camera_14, path_14, camera_15, path_15, max_depth=max_z_train, thres=thres_train*1.2).save_train_dataset()
positive_sample_test = CreatePositiveDataSet(camera_14, path_14, camera_16, path_16,max_depth=max_z_test, thres=thres_test).save_test_dataset()