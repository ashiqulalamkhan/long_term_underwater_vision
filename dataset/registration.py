import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir",default="/home/turin/Desktop/lizard_dataset_curated/2014/pcd14.pcd" , type=str,
                        help="Source Pcd Directory")
    parser.add_argument("--target_dir", default="/home/turin/Desktop/lizard_dataset_curated/2015/pcd15.pcd", type=str,
                        help="Target Pcd Directory")
    parser.add_argument("registration", type=str, help="Either 'fast', 'global' or 'icp'")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="User Defined Voxel_size, initial value = .1m")
    parser.add_argument("--distance_threshold_mult", type=float, default=1.5, help="distance_threshold multiplier to "
                                                                                   "voxel size")
    parser.add_argument("--noise", help="With or without noise to source pcd, default False", default=False)
    parser.add_argument("--init_icp", default=o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32), help='o3d.core.Tensor 4*4'
                                                                                                   'Initialization Mat')
    parser.add_argument("--max_iter", type=int, default=50, help="Iteration Number")
    parser.add_argument("--max_corr_dist", type=float, default=.8, help="ICP max_correspondence_distances ")
    parser.add_argument("--hist", type=str, default="True", help="Histogram 'True','False' default:'True'")
    parser.add_argument("--hist_bins", type=int, default=1000, help="Histogram bin Number")
    args = parser.parse_args()
    return args


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def registration_histogram(source, target, params):
    dist_error = source.compute_point_cloud_distance(target)
    error = []
    for x in range(len(dist_error)):
        error.append(dist_error[x])
    print("Maximum Error:", max(error), " Minimum Error: ", min(error), " Average Error:", sum(error) / len(error))
    if params.hist == "True":
        print("Histogram normal scale")
        hist, bin = np.histogram(error, params.hist_bins)
        plt.bar(bin[:-1] + np.diff(bin) / 2, hist, np.diff(bin))
        plt.xlabel("Error in meters(for 1000 bins)")
        plt.ylabel("number of cloud point")
        plt.show()
        print("Histogram Log Scale")
        plt.bar(bin[:-1] + np.diff(bin) / 2, hist, np.diff(bin))
        plt.xlabel("Error in meters(for 1000 bins)")
        plt.ylabel("number of cloud point in Log scale")
        plt.yscale("log")
        plt.show()

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 3
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds with initial pose.")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_fast_reg(source, target, params):
    # FAST GLOBAL REGISTRATION
    draw_registration_result(source, target, np.identity(4))
    source_cp = copy.deepcopy(source)
    target_cp = copy.deepcopy(target)
    start = time.time()
    voxel_size = params.voxel_size  # 0.05 means 5cm for this dataset
    distance_threshold = voxel_size * 1.5
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_cp, target_cp,
                                                                                         voxel_size)
    print(":: Apply fast global registration with distance threshold %.3f"
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result)
    draw_registration_result(source, target, result.transformation)
    source_cp.transform(result.transformation)
    registration_histogram(source_cp, target_cp, params)
    return result.transformation

def execute_global_reg(source, target, params):
    #GLOBAL REGISTRATION
    draw_registration_result(source, target, np.identity(4))
    source_cp = copy.deepcopy(source)
    target_cp = copy.deepcopy(target)
    start = time.time()
    voxel_size = params.voxel_size  # 0.05 means 5cm for this dataset
    distance_threshold = voxel_size * 2.5
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_cp, target_cp,
                                                                                         voxel_size)
    print(":: Apply fast global registration with distance threshold %.3f"
          % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
        3, criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 1.0))
    print("Global registration took %.3f sec.\n" % (time.time() - start))
    print(result)
    draw_registration_result(source, target, result.transformation)
    source_cp.transform(result.transformation)
    registration_histogram(source_cp, target_cp, params)
    return result.transformation




def execute_local_icp_reg(source, target, params):
    # 'Returns np 4*4 transformation matrix'
    source_cp = source.clone()
    target_cp = target.clone()
    source_leg = source_cp.to_legacy()
    target_leg = target_cp.to_legacy()
    draw_registration_result(source_leg, target_leg, np.identity(4))
    start = time.time()
    treg = o3d.t.pipelines.registration
    voxel_sizes = o3d.utility.DoubleVector([params.voxel_size, params.voxel_size * .5, params.voxel_size * .25])
    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                    relative_rmse=0.000001,
                                    max_iteration= 200),#params.max_iter),
        treg.ICPConvergenceCriteria(0.0000001, 0.0000001, int(params.max_iter*0.5)),
        treg.ICPConvergenceCriteria(0.00000001, 0.00000001, int(params.max_iter*0.25))
    ]
    max_correspondence_distances = o3d.utility.DoubleVector(
        [params.max_corr_dist, params.max_corr_dist * 0.9, params.max_corr_dist * 0.8])
    estimation = treg.TransformationEstimationPointToPlane()

    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    callback_after_iteration = lambda loss_log_map: print(
        "Iteration Index: {}, Scale Index: {}, Scale Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
            loss_log_map["iteration_index"].item(),
            loss_log_map["scale_index"].item(),
            loss_log_map["scale_iteration_index"].item(),
            loss_log_map["fitness"].item(),
            loss_log_map["inlier_rmse"].item()))
    registration_ms_icp = treg.multi_scale_icp(source_cp, target_cp, voxel_sizes,
                                               criteria_list,
                                               max_correspondence_distances, params.init_icp,
                                               estimation, callback_after_iteration)

    print("Inlier Fitness: ", registration_ms_icp.fitness)
    print("Inlier RMSE: ", registration_ms_icp.inlier_rmse)
    print("ICP Local registration took %.3f sec.\n" % (time.time() - start))
    # source_leg = source.to_legacy()
    # target_leg = target.to_legacy()
    reg = registration_ms_icp.transformation.numpy()
    draw_registration_result(source_leg, target_leg, reg)
    source_leg.transform(reg)
    registration_histogram(source_leg, target_leg, params)
    return registration_ms_icp.transformation.numpy()


def main():
    r = R.from_rotvec([np.pi / 9, 0, np.pi / 6])
    rot_mat = r.as_matrix().reshape(3, 3)
    params = parse_args()
    source = o3d.io.read_point_cloud(params.source_dir)
    target = o3d.io.read_point_cloud(params.target_dir)
    if params.noise:
        print("Introducing Noise")
        source.translate(np.asarray([1.12, 1.55, -1.1]))
        source.rotate(rot_mat, center=source.get_center())
        source.scale(scale=1.5, center=source.get_center())
    if params.registration == "global":
        print("Executing Global Registration")
        global_trans_mat = execute_global_reg(source, target, params)
        print("Final Transformation Matrix", global_trans_mat)
    if params.registration == "fast":
        print("Executing Fast Global Registration")
        fast_trans_mat = execute_global_fast_reg(source, target, params)
        print("Final Transformation Matrix", fast_trans_mat)
    if params.registration == "icp":
        device = o3d.core.Device("CPU:0")
        source_t = o3d.t.geometry.PointCloud(device)
        target_t = o3d.t.geometry.PointCloud(device)
        source_t = source_t.from_legacy(source)
        target_t = target_t.from_legacy(target)
        print("Executing ICP Registration")
        icp_trans_mat = execute_local_icp_reg(source_t, target_t, params)
        print("Final Transformation Matrix", icp_trans_mat)


if __name__ == "__main__":
    main()
