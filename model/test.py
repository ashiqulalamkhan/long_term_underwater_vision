import numpy as np
import long_term_underwater_vision.dataset.registration as rgs
import vis_camera
import camera



#######
camera14_xml = "/home/turin/Desktop/lizard_dataset_curated/2014/cam14.xml"
camera15_xml = "/home/turin/Desktop/lizard_dataset_curated/2015/cam15.xml"
opencv_xml = "/home/turin/Desktop/lizard_dataset_curated/opencv_cam_calib.xml"

####
camera_set14 = camera.main(camera14_xml, opencv_xml)
camera_set15 = camera.main(camera15_xml, opencv_xml)
labels = ["camera 14","camera 15"]
vis_camera.main([camera_set14, camera_set15], labels, True)
####

camera_xmls = [camera14_xml, camera15_xml]
opencv_xmls = [opencv_xml, opencv_xml]
labels = ["camera 14","camera 15"]
vis_camera.main(camera_xmls, opencv_xmls, labels, True)

#POINT CLOUD REGISTRATION
params = rgs.Params()
params.source_dir = "/home/turin/Desktop/lizard_dataset_curated/2015/pcd15.pcd"
params.target_dir = "/home/turin/Desktop/lizard_dataset_curated/2014/pcd14.pcd"
trans_mat = rgs.main(params)
shift = np.asarray([332400, 8375600, 0.0])
#CAMERA FILES
camera14_xml = "/home/turin/Desktop/lizard_dataset_curated/2014/cam14.xml"
camera15_xml = "/home/turin/Desktop/lizard_dataset_curated/2015/cam15.xml"
opencv_xml = "/home/turin/Desktop/lizard_dataset_curated/opencv_cam_calib.xml"
#FOR CAMERA14
camera_ref14 = camera.CameraRef(camera14_xml, opencv_xml)
camera14_set = []
for idx in range(len(camera_ref14.camera_dict)):
    camera14 = camera.Camera(camera_ref14.camera_dict[idx], camera_ref14.local2global, shift=shift)
    camera14_set.append(camera14.get_camera_set())

#FOR CAMERA15
camera_ref15 = camera.CameraRef(camera15_xml, opencv_xml)
camera15_set = []
for idx in range(len(camera_ref15.camera_dict)):
    camera15 = camera.Camera(camera_ref15.camera_dict[idx], camera_ref15.local2global, trans_mat, shift=shift)
    camera15_set.append(camera15.get_camera_set())

print("Done")




