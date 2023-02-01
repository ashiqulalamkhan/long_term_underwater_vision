import open3d as o3d

pcd_load = o3d.io.read_point_cloud("error.ply")
o3d.visualization.draw_geometries([pcd_load])


# frags = o3d.read_point_cloud("error.ply")
# visualizer = JVisualizer()
# visualizer.add_geometry(frags)
# visualizer.show()