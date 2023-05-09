import numpy as np
import vis_camera
import camera
import numpy as np
import os
import shutil
import preprocess_dataset
from pyproj import CRS
from pyproj import Transformer
import Metashape
import time
import threading
import multiprocessing
import concurrent.futures
import pickle


def metashape_descriptors(project_path):
    doc = Metashape.Document()
    doc.open(project_path)
    chunk = doc.chunk
    cameras = chunk.cameras
    #Get all tie points
    point_cloud = chunk.tie_points
    #Get camera projections of the tie points
    projections = point_cloud.projections
    #Get the points
    points = point_cloud.points
    #Sift descriptors
    desc_points = {}
    #Iterate over all camera
    for cam in cameras:
        desc_2D3D = {}
        sift_points = []
        #3D point per 2D sift points
        pcd_3d_points = []
        for proj in projections[cam]:
            #Append all 2D
            sift_points.append(proj.coord)
            #Append all 3D
            pcd_3d_points.append(cam.unproject(proj.coord))
        desc_2D3D["2D"] = sift_points
        desc_2D3D["3D"] = pcd_3d_points
        desc_points[cam.label]=desc_2D3D
    return desc_points

def desc3dlocal_to_global(camera_ref, num_worker = 8):
    start_step = 0
    step = (len(camera_ref.camera_dict)/num_worker)
    step_list = []
    for idx in range(num_worker):
        if idx==num_worker-1:
            step_list.append([int(start_step), int(len(camera_ref.camera_dict))])
        else:
            step_list.append([int(start_step), int(start_step+step)])
            start_step = start_step+step
    print("Total cameras",len(camera_ref.camera_dict),"Step List",step_list)
    return step_list

def run(start, end):
    xyz = {}
    for idx in range(start, end, 1):
        camera_cls = camera.Camera(camera_ref.camera_dict[idx], camera_ref.local2global, shift=shift)
        point2d = desc_points[camera_cls.camera_id]["2D"]
        point3d = desc_points[camera_cls.camera_id]["3D"]
        new_point3d= []
        for pts in point3d:
            xyz14 = np.hstack([np.asarray(pts), 1.0])
            xyz14global = camera_ref.local2global @ xyz14
            ####LOCAL TO GLOBAL
            xyz_crs4978 = xyz14global
            crs4978 = CRS.from_epsg(4978)
            crs32755 = CRS.from_epsg(32755)
            proj = Transformer.from_crs(crs4978, crs32755)
            xyz_crs4978[0,:3] = proj.transform(xyz_crs4978[0,0], xyz_crs4978[0,1], xyz_crs4978[0,2])
            xyz_crs4978 = np.asarray(xyz_crs4978).squeeze()
            xyz_crs4978[0] -= shift[0]
            xyz_crs4978[1] -= shift[1]
            xyz_crs4978[2] -= shift[2]
            xyz_transformed = transform@xyz_crs4978
            new_point3d.append(xyz_transformed)
        #desc_points[camera14.camera_id]["3D"] = new_point3d
        xyz[camera_cls.camera_id] = new_point3d
    return xyz
def multi_process():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(run, sec1, sec2) for sec1, sec2 in step_list]
    final_result = []
    i = 0
    for result in results:
        print(i)
        final_result.append(result.result())
        i+=1
    return final_result

transformed_camera_sets = preprocess_dataset.camera_transformation()
camera_14, camera_15, camera_16, cam14cls, cam15cls, cam16cls = transformed_camera_sets
camera_14 = np.asarray(camera_14, dtype="object")
camera_15 = np.asarray(camera_15, dtype="object")
camera_16 = np.asarray(camera_16, dtype="object")
cam14cls = np.asarray(cam14cls, dtype="object")
cam15cls = np.asarray(cam15cls, dtype="object")
cam16cls = np.asarray(cam16cls, dtype="object")
cam15cls[0].transformation_matrix, cam15cls[0].shift

camera14_xml = "/home/turin/Desktop/lizard_dataset_curated/2014/cam14.xml"
camera15_xml = "/home/turin/Desktop/lizard_dataset_curated/2015/cam15.xml"
camera16_xml = "/home/turin/Desktop/lizard_dataset_curated/2016/cam16.xml"
opencv_xml = "/home/turin/Desktop/lizard_dataset_curated/opencv_cam_calib.xml"
project_path14 = "/home/turin/Desktop/lizard_jackson/chronological/2014/r20141102_074952_lizard_d2_081_horseshoe_circle01/r20141102_074952_lizard_d2_081_horseshoe_circle01.psx"
project_path15 = "/home/turin/Desktop/lizard_island/jackson/chronological/2015/r20151207_222558_lizard_d2_039_horseshoe_circle01/r20151207_222558_lizard_d2_039_horseshoe_circle01.psx"
project_path16 = "/home/turin/Desktop/lizard_island/jackson/chronological/2016/r20161121_063027_lizard_d2_050_horseshoe_circle01/r20161121_063027_lizard_d2_050_horseshoe_circle01.psx"
camera_ref14 = camera.CameraRef(camera14_xml, opencv_xml)
camera_ref15 = camera.CameraRef(camera15_xml, opencv_xml)
camera_ref16 = camera.CameraRef(camera16_xml, opencv_xml)

desc14 = metashape_descriptors(project_path14)
desc15 = metashape_descriptors(project_path15)
desc16 = metashape_descriptors(project_path16)

# step_list14 = desc3dlocal_to_global(desc14, camera_ref14, 14)
# step_list15 = desc3dlocal_to_global(desc15, camera_ref15, 14)
# step_list16 = desc3dlocal_to_global(desc16, camera_ref16, 14)

print("STARTING 14")
camera_ref = camera_ref14
transform = cam14cls[0].transformation_matrix
shift = cam14cls[0].shift
desc_points = desc14
step_list = desc3dlocal_to_global(camera_ref, num_worker = 14)
print(f"For Camera Classs 14: step list: {step_list}, transform: {transform}, shift = {shift} ")
start = time.time()
point3d_global14 = multi_process()
end = time.time()
print("Ended 14 in:", str(end-start))
print("Saving 14")
data_test3d = open("data3d14.pickle", "wb")
pickle.dump(point3d_global14, data_test3d)
data_test3d.close()


print("STARTING 15")
camera_ref = camera_ref15
transform = cam15cls[0].transformation_matrix
shift = cam15cls[0].shift
desc_points = desc15
step_list = desc3dlocal_to_global(camera_ref, num_worker = 14)
print(f"For Camera Classs 15: step list: {step_list}, transform: {transform}, shift = {shift} ")
start = time.time()
point3d_global15 = multi_process()
end = time.time()
print("Ended 15 in:", str(end-start))
print("Saving 15")
data_test3d = open("data3d15.pickle", "wb")
pickle.dump(point3d_global15, data_test3d)
data_test3d.close()


print("STARTING 16")
camera_ref = camera_ref16
transform = cam16cls[0].transformation_matrix
shift = cam16cls[0].shift
desc_points = desc16
step_list = desc3dlocal_to_global(camera_ref, num_worker = 14)
print(f"For Camera Classs 16: step list: {step_list}, transform: {transform}, shift = {shift} ")
start = time.time()
point3d_global16 = multi_process()
end = time.time()
print("Ended 16 in:", str(end-start))
print("Saving 16")
data_test3d = open("data3d16.pickle", "wb")
pickle.dump(point3d_global16, data_test3d)
data_test3d.close()

print("WORKED")

