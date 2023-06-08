import open3d as o3d
import numpy as np
from PIL import Image
import os
import csv
import argparse

def pcd_to_heightmap_patches(pcd_file, name, patch_size_m=5.0, img_size_px=28):
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    min_y, max_y = points[:, 1].min(), points[:, 1].max()

    patch_num_x = int((max_x - min_x) / patch_size_m) + 1
    patch_num_y = int((max_y - min_y) / patch_size_m) + 1

    full_image = np.zeros((img_size_px * patch_num_y, img_size_px * patch_num_x), dtype=np.uint8)

    patch_info = [['patch', 'min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']]

    for i in range(patch_num_y):
        for j in range(patch_num_x):
            patch_min_x = min_x + j * patch_size_m
            patch_max_x = min_x + (j + 1) * patch_size_m
            patch_min_y = min_y + i * patch_size_m
            patch_max_y = min_y + (i + 1) * patch_size_m

            points_in_patch = points[(points[:, 0] >= patch_min_x) & (points[:, 0] < patch_max_x) &
                                     (points[:, 1] >= patch_min_y) & (points[:, 1] < patch_max_y)]

            if len(points_in_patch) > 0:
                # Rescale z-values between 0 and 255
                min_z = points_in_patch[:, 2].min()
                max_z = points_in_patch[:, 2].max()
                scaled_z = (points_in_patch[:, 2] - min_z) / (max_z - min_z) * 255

                pixel_x = ((points_in_patch[:, 0] - patch_min_x) / patch_size_m * (img_size_px - 1)).astype(int)
                pixel_y = ((points_in_patch[:, 1] - patch_min_y) / patch_size_m * (img_size_px - 1)).astype(int)

                heightmap = np.zeros((img_size_px, img_size_px), dtype=np.uint8)
                heightmap[pixel_y, pixel_x] = scaled_z.astype(np.uint8)

                # Save individual patch
                patch_folder = f'./{name}'
                if not os.path.exists(patch_folder):
                    os.makedirs(patch_folder)
                patch_filename = f'patch_{i}_{j}.png'
                Image.fromarray(heightmap).save(os.path.join(patch_folder, patch_filename))

                full_image[i * img_size_px:(i + 1) * img_size_px, j * img_size_px:(j + 1) * img_size_px] = heightmap

                patch_info.append([patch_filename, patch_min_x, patch_max_x, patch_min_y, patch_max_y, min_z, max_z])

    Image.fromarray(full_image).save(f'./{name}_whole_image.png')

    with open(f'./{name}.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(patch_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heightmap patches from a PCD file.')
    parser.add_argument('pcd_file', type=str, help='Path to the PCD file.')
    parser.add_argument('name', type=str, help='The name of result.')
    parser.add_argument('patch_size_m', type=float, default=3.0, help='Patch size in meters.')
    parser.add_argument('img_size_px', type=int, default=15, help='Image size in pixels.')
    args = parser.parse_args()

    pcd_to_heightmap_patches(args.pcd_file, args.name, args.patch_size_m, args.img_size_px)
