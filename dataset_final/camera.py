import argparse
import numpy as np
import xml.etree.ElementTree as ET
from pyproj import CRS
from pyproj import Transformer

"""
HERE WE ARE CONCERNED WITH LOADING CAMERA XML FILES BOTH INTRINSICS AND EXTRINSIC. 
IN ADDITION IMPORTANT CLASS METHODS.
GENERAL WORKFLOW FOR US WILL INVOLVE:

1. Load Camera Intrinsics and Extrinsic using "CameraRef" Class
2. Use "Camera" Class to: 
    a. Convert camera pose from internal coordinate to global coordinates 
    b. Convert these global pose with required transformation
    c. Return a camera_set with [image_id, camera_pose, reference]
3. Class for Relative pose Estimation

"""
class CameraRef:
    def __init__(self, camera_xml, calib_xml_opencv):
        """Input takes:
        1. camera_xml file from: metashape->File->Export->Export Cameras
        2. calib_xml file from: metashape->Tools->Camera Calibration->Adjusted->Save As(OpenCv Format)
        """
        self.root = ET.parse(camera_xml).getroot()
        self.camera_xml = camera_xml
        self.calib_xml_opencv = calib_xml_opencv
        self.camera_calibration_mat = self.get_camera_calib()
        self.camera_calibration_mat_opencv = self.get_camera_calib_opencv()
        self.camera_dict = self.get_cam_dict()
        self.local2global = self.get_local2global_matrix()

    def get_local2global_matrix(self):
        root = self.root
        rotation = np.matrix(root[0][3][0].text).reshape(3, 3)
        translation = np.matrix(root[0][3][1].text).reshape(1, 3)
        scale = float(root[0][3][2].text)
        rt = np.vstack([np.hstack([scale * rotation, translation.T]), [0, 0, 0, 1]])
        return rt

    def get_cam_dict(self):
        root = self.root
        camera = []
        for cam in root.iter("camera"):
            # camera_id = int(list(cam.attrib.values())[0])
            img_id = list(cam.attrib.values())[-1]
            for item in cam.iter("transform"):
                val = np.matrix(item.text)
                transform_mat = np.reshape(val, (4, 4))
            for item in cam.iter("reference"):
                ref = []
                for idx in item.attrib.values():
                    if idx != "true":
                        ref.append(float(idx))
                ref = np.asarray(ref)
            camera_dict = {"image_id": img_id,
                           "transform_mat": transform_mat,
                           "reference": ref}
            camera.append(camera_dict)
        return camera

    def get_camera_calib(self):
        root = self.root
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

    def get_camera_calib_opencv(self):
        tree = ET.parse(self.calib_xml_opencv)
        root = tree.getroot()
        for x in root.iter("Camera_Matrix"):
            for y in x.iter("data"):
                camera_mat = np.matrix(y.text)
        for x in root.iter("Distortion_Coefficients"):
            for y in x.iter("data"):
                distort_mat = np.matrix(y.text)
        camera_mat = np.asarray(camera_mat.reshape((3, 3)))
        distort_mat = np.asarray(distort_mat)
        # print(camera_mat, distort_mat)
        return [camera_mat, distort_mat]


class Camera:
    """
    Camera class inputs:
    1. Camera Dict single camera not a list
    2. local2global transformation mat from CameraRef Class
    It stores:
    1. Camera Id/Image name
    2. Pose matrix in internal metashape coord
    3. reference on global coord
    4. method for Pose in global coord
    5. Also Transformation
    """
    def __init__(self, camera_dict, local2global, transformation_matrix=np.eye(4), shift=np.asarray([0.0,0.0,0.0])):
        self.camera_id = camera_dict["image_id"]
        self.pose = camera_dict["transform_mat"]
        self.reference = camera_dict["reference"]
        self.local2global_transmat = local2global
        self.transformation_matrix = transformation_matrix
        self.shift = shift
        self.global_pose = self.get_global_pose()
        self.global_pose_transformed = self.transform()

    def get_global_pose(self):
        pose_crs4978 = self.local2global_transmat * self.pose
        crs4978 = CRS.from_epsg(4978)
        crs32755 = CRS.from_epsg(32755)
        proj = Transformer.from_crs(crs4978, crs32755)
        pose_crs4978[0, 3], pose_crs4978[1, 3], pose_crs4978[2, 3] = proj.transform(pose_crs4978[0, 3],
                                                                                    pose_crs4978[1, 3],
                                                                                    pose_crs4978[2, 3])
        pose_crs32755 = pose_crs4978
        return pose_crs32755

    def transform(self):
        transform_pose = self.global_pose
        transform_pose[0, 3] -= self.shift[0]
        transform_pose[1, 3] -= self.shift[1]
        transform_pose[2, 3] -= self.shift[2]
        return self.transformation_matrix @ transform_pose

    def get_camera_set(self):
        return [self.camera_id, self.global_pose_transformed]


class RelativePose:
    def __init__(self, mat1, mat2):
        self.mat1 = mat1
        self.mat2 = mat2
        self.Rt21 = self.mat2r_t(mat1, mat2)
        self.Rt12 = self.mat2r_t(mat2, mat1)

        def mat2r_t(mat1, mat2):
            U, S, Vt = np.linalg.svd(np.dot(mat2.T, mat1))

            R = np.dot(U, Vt)
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            org = np.dot(np.linalg.inv(mat2), mat1)
            t = org[:3, 3] / np.linalg.norm(org[:3, 3])
            return R, t


# class CameraTransform:
#     def __init__(self, camera_set, transformation_matrix):
#         self.camera_set = camera_set
#         self.transformation_matrix = transformation_matrix
#         self.camera_set_transformed = self.camera_transform(camera_set, transformation_matrix)
#
#     def camera_transform(self):
#         camera_transformed = self.camera_set
#         for idx in range(len(self.camera_set)):
#             camera_transformed[idx][1] = self.transformation_matrix@camera_set[idx][1]
#         return camera_transformed

def main(camera_xml, camera_opencv_xml):
    camera_ref = CameraRef(camera_xml, camera_opencv_xml)
    camera_set = []
    for idx in range(len(camera_ref.camera_dict)):
        camera = Camera(camera_ref.camera_dict[idx], camera_ref.local2global)
        camera_pose = camera.get_camera_set()
        camera_set.append(camera_pose)
    return camera_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_xml", default="/home/turin/Desktop/lizard_dataset_curated/2014/cam14.xml", type=str,
                        help="Camera XML File")
    parser.add_argument("--opencv_xml", default="/home/turin/Desktop/lizard_dataset_curated/opencv_cam_calib.xml",
                        type=str, help="OpenCV Camera CalibrationXML file")
    args = parser.parse_args()
    camera_set = main(args.camera_xml, args.opencv_xml)
