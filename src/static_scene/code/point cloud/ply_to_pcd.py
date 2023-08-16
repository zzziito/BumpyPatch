import open3d as o3d

def convert_ply_to_pcd(ply_file_path, pcd_file_path):
    # Load the ply file
    ply = o3d.io.read_point_cloud(ply_file_path)
    
    # Write point cloud to a pcd file
    o3d.io.write_point_cloud(pcd_file_path, ply)
    print(f"Converted {ply_file_path} to {pcd_file_path}")

# Example usage
convert_ply_to_pcd("/media/rtlink/JetsonSSD-256/Download/Rellis_3D_lidar_example/os1_cloud_node_color_ply/frame000007-1581624653_470.ply", "./sample1.pcd")
