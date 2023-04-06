import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import pickle


# CAMERA INTRINSICS FROM METASHAPE xml FILE
def get_cam_intrinsics(root):
    camera_intrinsics = []
    y = []
    for node in root.iter("calibration"):
        x = list(node.attrib.values())
        if (x[1] == "adjusted"):
            for node2 in node.iter():
                camera_intrinsics.append(node2.text)
                y.append(list(node2.attrib.values()))
    camera_intrinsics = list(map(float, camera_intrinsics[2:]))
    width, height = list(map(float, y[1]))
    camera_intrinsics_dict = {"width": width,
                              "height": height,
                              "f": camera_intrinsics[0],
                              "cx": camera_intrinsics[1],
                              "cy": camera_intrinsics[2],
                              "b1": camera_intrinsics[3],
                              "b2": camera_intrinsics[4],
                              "k1": camera_intrinsics[5],
                              "k2": camera_intrinsics[6],
                              "k3": camera_intrinsics[7],
                              "k4": camera_intrinsics[8],
                              "p1": camera_intrinsics[9],
                              "p2": camera_intrinsics[10]}
    return camera_intrinsics_dict


# CAMERA POSE PER IMAGE ID
def get_cam_pose(root):
    camera = []
    for cam in root.iter("camera"):
        camera_id = int(list(cam.attrib.values())[0])
        img_id = list(cam.attrib.values())[-1]
        # print(camera_id, img_id)
        for item in cam.iter("transform"):
            val = np.matrix(item.text)
            transform_mat = np.reshape(val, (4, 4))
            # print(transform_mat)
        for item in cam.iter("reference"):
            ref = item.attrib
            # print(ref)
        camera_dict = {"image_id": img_id,
                       "transform_mat": transform_mat,
                       "reference": ref}
        camera.append(camera_dict)
    camera_pose = camera
    pose = []
    for idx in range(len(camera_pose)):
        cam_ref = [0, 0, 0, 0, 0, 0]
        cam_id = camera_pose[idx]["image_id"]
        cam_pose = camera_pose[idx]["transform_mat"]
        cam_ref[0] = float(camera_pose[idx]["reference"]["x"]) - 332400
        cam_ref[1] = float(camera_pose[idx]["reference"]["y"]) - 8375600
        cam_ref[2] = float(camera_pose[idx]["reference"]["z"])
        cam_ref[3] = float(camera_pose[idx]["reference"]["yaw"])
        cam_ref[4] = float(camera_pose[idx]["reference"]["pitch"])
        cam_ref[5] = float(camera_pose[idx]["reference"]["roll"])
        pose.append([cam_id, cam_pose, cam_ref])
    im_pose = np.asarray(pose)
    # file = open('image_id', 'wb')
    # pickle.dump(pose, file)
    # file.close()
    return im_pose

def reference_grouping(pose1, pose2):
    # BASED ON REFERENCE
    d_threshold = .2
    neg_threshold = 2
    dict_group = {}
    neg_group_reduced = []
    for idx in range(len(pose1)):
        group = []
        neg_group = []
        for idy in range(len(pose2)):
            mat1 = pose1[idx][2]
            mat2 = pose2[idy][2]
            if mat1[0] - d_threshold < mat2[0] < mat1[0] + d_threshold and mat1[1] - d_threshold < mat2[1] < mat1[
                1] + d_threshold:  # and mat1[2]-thres<mat2[2]<mat1[2]+thres:
                group.append(pose2[idy])
            if (mat2[0] < mat1[0] - neg_threshold or mat2[0] > mat1[0] + neg_threshold) and (
                    mat2[1] < mat1[1] - neg_threshold or mat2[1] > mat1[1] + neg_threshold):
                neg_group.append(pose2[idy])
        # print(len(group))
        # print(len(neg_group))
        if len(neg_group) > 2:
            neg_group_reduced = neg_group[:len(group)]
        dict_group[pose1[idx][0]] = [group, neg_group_reduced]
    return dict_group

def transmat_grouping(pose1, pose2):
    thres = 2
    neg_thres = 60
    dict_group = {}
    for idx in range(len(pose1)):
        group = []
        neg_group = []
        trans_mat = np.asarray([[ 0.93188237,  0.00785442,  0.02757995,  1.27660544],
       [-0.00684264,  0.93167355, -0.03412714,  3.52793686],
       [-0.02784823,  0.03390857,  0.93129038,  0.49316502],
       [ 0.0      ,  0.0        ,  0.0        ,  1.0        ]])
        for idy in range(len(pose2)):
            mat1 = pose1[idx][1]
            mat2 = pose2[idy][1]
            if mat1[0,3]-thres<mat2[0,3]<mat1[0,3]+thres and mat1[1,3]-thres<mat2[1,3]<mat1[1,3]+thres and mat1[2,3]-thres<mat2[2,3]<mat1[2,3]+thres:
                group.append(pose2[idy])
            if (mat2[0,3] < mat1[0,3]-neg_thres or mat2[0,3]>mat1[0,3]+neg_thres) and (mat2[1,3] < mat1[1,3]-neg_thres or mat2[1,3]>mat1[1,3]+neg_thres):
                neg_group.append(pose2[idy])
        if len(neg_group)>2:
            neg_group_reduced = neg_group[:len(group)]
        dict_group[pose1[idx][0]] = [group, neg_group_reduced]


def main():
    file1 = "/home/turin/Desktop/lizard_dataset_curated/2014/cam14.xml"
    file2 = "/home/turin/Desktop/lizard_dataset_curated/2015/cam15.xml"
    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()
    root2 = tree2.getroot()
    camera_intrinsics1 = get_cam_intrinsics(root1)
    camera_intrinsics2 = get_cam_intrinsics(root2)
    # WE CALCULATE POSE ARRAY OF ALL CAMERA WITH  shape [Camera ID, Transformation Matrix, Geo Reference Vector]
    pose1 = get_cam_pose(root1)
    pose2 = get_cam_pose(root2)
    dict_group = transmat_grouping(pose1, pose2)
    import os
    import shutil
    # file1 = open('camera_group', 'rb')
    # camera_group = pickle.load(file1)
    # file1.close
    camera_group = dict_group
    path1 = "/home/turin/Desktop/lizard_island/jackson/chronological/2014/r20141102_074952_lizard_d2_081_horseshoe_circle01/081_photos/"
    path2 = "/home/turin/Desktop/lizard_island/jackson/chronological/2015/r20151207_222558_lizard_d2_039_horseshoe_circle01/039_photos/"
    for idx in (os.listdir(path1)):
        new_path = "/home/turin/Desktop/lizard_dataset_curated/dataset_group/" + idx[:-4]
        os.mkdir(new_path)
        for idy in camera_group[idx][0]:
            # print(idy[0])
            org_path = path2 + idy[0]
            shutil.copy(org_path, new_path)

    print("DONE")
    # file = open('/home/turin/Documents/GitHub/long_term_underwater_vision/dataset/camera_itrs', 'wb')
    # pickle.dump(camera_intrinsics, file)
    # file.close()


if __name__ == "__main__":
    main()
