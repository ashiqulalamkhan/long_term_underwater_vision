import numpy as np
import registration as rgs
import vis_camera
import camera
"""
BLOCK FOR POINT CLOUD REGISTRATION AND CAMERA REGISTRATION
"""
def camera_transformation(visualize=False):
    ####### REGISTRATION
    print("Registration started")
    #POINT CLOUD REGISTRATION 15 to 14
    params = rgs.Params()
    params.hist_name = "Hist_rmse_15_14"
    params.source_dir = "/home/turin/Desktop/lizard_dataset_curated/2015/pcd15.pcd"
    params.target_dir = "/home/turin/Desktop/lizard_dataset_curated/2014/pcd14.pcd"
    trans_mat_15_14 = rgs.main(params)
    #POINT CLOUD REGISTRATION 16 to 14
    params = rgs.Params()
    params.hist_name = "Hist_rmse_16_14"
    params.source_dir = "/home/turin/Desktop/lizard_dataset_curated/2016/pcd16.pcd"
    params.target_dir = "/home/turin/Desktop/lizard_dataset_curated/2014/pcd14.pcd"
    trans_mat_16_14 = rgs.main(params)
    print("Registration DONE")

    #VISUALIZATION
    #GLOBAL SHIFT ACCORDING TO POINTCLOUD SHIFT USED IN THE APP WHILE EXPORTING PCDS
    shift = np.asarray([332400, 8375600, 0.0])
    #CAMERA FILES
    camera14_xml = "/home/turin/Desktop/lizard_dataset_curated/2014/cam14.xml"
    camera15_xml = "/home/turin/Desktop/lizard_dataset_curated/2015/cam15.xml"
    camera16_xml = "/home/turin/Desktop/lizard_dataset_curated/2016/cam16.xml"
    opencv_xml = "/home/turin/Desktop/lizard_dataset_curated/opencv_cam_calib.xml"
    #FOR CAMERA14
    camera_ref14 = camera.CameraRef(camera14_xml, opencv_xml)
    camera14_set = []
    camera14_class = []
    for idx in range(len(camera_ref14.camera_dict)):
        camera14 = camera.Camera(camera_ref14.camera_dict[idx], camera_ref14.local2global, shift=shift)
        camera14_set.append(camera14.get_camera_set())
        camera14_class.append(camera14)

    #FOR CAMERA15
    camera_ref15 = camera.CameraRef(camera15_xml, opencv_xml)
    camera15_set = []
    camera15_class = []
    for idx in range(len(camera_ref15.camera_dict)):
        camera15 = camera.Camera(camera_ref15.camera_dict[idx], camera_ref15.local2global, trans_mat_15_14, shift=shift)
        camera15_set.append(camera15.get_camera_set())
        camera15_class.append(camera15)

    #FOR CAMERA16
    camera_ref16 = camera.CameraRef(camera16_xml, opencv_xml)
    camera16_set = []
    camera16_class = []
    for idx in range(len(camera_ref16.camera_dict)):
        camera16 = camera.Camera(camera_ref16.camera_dict[idx], camera_ref16.local2global, trans_mat_16_14, shift=shift)
        camera16_set.append(camera16.get_camera_set())
        camera16_class.append(camera16)

    print("CAMERA DATA Preproccesing Done")
    if visualize:
        #INITIAL POINCLOUD:
        labels = ["camera 14", "camera 15", "camera 16"]
        camera14 = camera.main(camera14_xml, opencv_xml)
        camera15 = camera.main(camera15_xml, opencv_xml)
        camera16 = camera.main(camera16_xml, opencv_xml)
        xyz14 = vis_camera.pose2xyz(camera14, True)
        xyz15 = vis_camera.pose2xyz(camera15, True)
        xyz16 = vis_camera.pose2xyz(camera16, True)
        camera_xyzs = [xyz14, xyz15, xyz16]
        for x in range(len(camera_xyzs)):
            vis_camera.visualize_camera(camera_xyzs[x], "3d", "_init_"+labels[x])
            vis_camera.visualize_camera(camera_xyzs[x], "2d", "_init_"+labels[x])
        vis_camera.visualize_multi_camera(camera_xyzs, labels, "2d", "_init_multi")
        vis_camera.visualize_multi_camera(camera_xyzs, labels, "3d", "_init_multi")

        #REGISTERED POINT CLOUD
        camera_sets = [camera14_set, camera15_set, camera16_set]
        labels = ["Transformed camera 14", "Transformed camera 15", "Transformed camera 16"]
        vis_camera.main(camera_sets, labels, False)
    return camera14_set, camera15_set, camera16_set, camera14_class, camera15_class, camera16_class

