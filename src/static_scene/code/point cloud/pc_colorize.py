import argparse
import open3d as o3d
import pandas as pd
import numpy as np

def colorize_pcd(pcd_file, csv_file, output_file):
    # Read PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Define colors for each class in RGB format
    color_dict = {1: [0, 0, 1],  # blue
                  2: [0, 1, 0],  # green
                  3: [1, 0.5, 0],  # orange
                  4: [1, 0, 0]}  

    patch_info = pd.read_csv(csv_file)

    # Iterate over each patch
    for i, row in patch_info.iterrows():
        min_x, max_x = row['min_x'], row['max_x']
        min_y, max_y = row['min_y'], row['max_y']
        class_ = row['class']

        # Find the points within the patch
        mask = ((points[:, 0] >= min_x) & (points[:, 0] <= max_x) & 
                (points[:, 1] >= min_y) & (points[:, 1] <= max_y))

        # Change the color of the points within the patch
        colors[mask] = color_dict[class_]

    # Update the colors of the point cloud and save it
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_file, pcd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize point cloud based on CSV file')
    parser.add_argument('pcd_file', help='Input PCD file')
    parser.add_argument('csv_file', help='CSV file containing patch info')
    parser.add_argument('output_file', help='Output PCD file')
    args = parser.parse_args()

    colorize_pcd(args.pcd_file, args.csv_file, args.output_file)
