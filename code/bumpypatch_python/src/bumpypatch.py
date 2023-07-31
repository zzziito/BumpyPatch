import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import cv2
import math
import time

patches_info = []
ring_widths = []
interval = 5  # image save interval in seconds
points_threshold = 20  # configurable threshold for minimum points
last_save_time = time.time()


def point_cloud_callback(cloud_msg):
    rospy.loginfo("inside pc callback")
    global last_save_time
    gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)

    points = np.array(list(gen))
    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)

    min_z, max_z = np.min(points[:, 2]), np.max(points[:, 2])

    patches = divide_into_patches(point_cloud_o3d)
    rospy.loginfo("after divide into patches")

    for patch in patches:
        # skip if the number of points is less than threshold
        if len(patch.points) < points_threshold:
            continue

        patch_info = get_patch_info(patch)
        patches_info.append(patch_info)

        heightmap = get_heightmap(patch, min_z, max_z)
        # Store the heightmap image of the patch ...

    # Save the patch images every 5 seconds
    if time.time() - last_save_time >= interval:
        save_patches_as_image(patches_info)
        last_save_time = time.time()


def divide_into_patches(cloud, angle_res=10, distance_res=1.0):
    rospy.loginfo("inside divide into patches")
    # Convert cartesian to polar coordinates
    xy = np.asarray(cloud.points)[:, :2]
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    radii = np.sqrt(np.sum(np.square(xy), axis=1))

    # Define patches
    patches = {}
    for i, angle in enumerate(angles):
        for j, radius in enumerate(radii):
            patch_angle = math.floor(angle / angle_res) * angle_res
            patch_radius = math.floor(radius / distance_res) * distance_res

            patch_key = (patch_angle, patch_radius)
            if patch_key not in patches:
                patches[patch_key] = []

            patches[patch_key].append(cloud.points[i])

    # Convert patches into open3d point cloud objects
    for key in patches:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(patches[key]))
        patches[key] = point_cloud

    # Remove patches with insufficient points
    patches = {k: v for k, v in patches.items() if len(v.points) >= points_threshold}

    return list(patches.values())


def get_patch_info(patch):
    rospy.loginfo("inside get patch info")
    points = np.asarray(patch.points)
    min_z = np.min(points[:, 2])
    max_z = np.max(points[:, 2])
    center = np.mean(points[:, :2], axis=0)
    num_points = len(points)

    patch_info = {
        "min_z": min_z,
        "max_z": max_z,
        "center": center,
        "num_points": num_points
    }
    
    return patch_info


def get_heightmap(patch, min_z, max_z):
    points = np.asarray(patch.points)

    # Normalize the z coordinates to the range [0, 255]
    z_values = points[:, 2]
    z_values_normalized = ((z_values - min_z) / (max_z - min_z)) * 255

    # Create a blank grayscale image
    size = int(np.ceil(np.sqrt(len(z_values_normalized))))  # square image size
    heightmap = np.zeros((size, size), np.uint8)

    # Set the intensity of each pixel based on the corresponding point's z value
    for i in range(size):
        for j in range(size):
            if i * size + j < len(z_values_normalized):
                heightmap[i, j] = z_values_normalized[i * size + j]

    return heightmap


def save_patches_as_image(patches_info, filename):
    # Determine the size of the output image
    max_x = max(info['center'][0] + info['heightmap'].shape[0] // 2 for info in patches_info)
    max_y = max(info['center'][1] + info['heightmap'].shape[1] // 2 for info in patches_info)

    # Create a blank image
    image = np.zeros((max_y + 1, max_x + 1), np.uint8)

    # Copy each patch's heightmap to the corresponding location in the image
    for info in patches_info:
        x = info['center'][0]
        y = info['center'][1]
        heightmap = info['heightmap']
        image[y:y+heightmap.shape[0], x:x+heightmap.shape[1]] = heightmap

    # Save the image
    cv2.imwrite(filename, image)
    rospy.loginfo("Image saved: %s", filename)


if __name__ == '__main__':
    rospy.init_node('points_subscriber_node')
    rospy.loginfo("node initialized.")
    rospy.Subscriber("/velodyne_points", PointCloud2, point_cloud_callback)

    rospy.spin()
