#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import PolygonArray
from geometry_msgs.msg import Polygon, Point32, PolygonStamped
import numpy as np
from math import sin, cos
import math
import struct
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import std_msgs
import tf

### DEFINE CONSTANTS ###


MIN_POINTS = 100

min_range_ = 3
min_range_z2_ = 12.3625
min_range_z3_ = 22.025
min_range_z4_ = 41.35
max_range_ = 100.0

min_ranges_   = [min_range_, min_range_z2_, min_range_z3_, min_range_z4_]

sector_const = 32

# num_zones = 4
# num_sectors_each_zone = [sector_const, sector_const ,sector_const, sector_const]
# num_rings_each_zone = [4, 4, 4, 4]
# min_ranges_each_zone = [2.7, 12.3625, 22.025, 41.35]

num_zones = 4
num_sectors_each_zone = [16, 32 ,54, 32]
num_rings_each_zone = [2, 4, 4, 4]

ring_sizes_   = [(min_range_z2_ - min_range_) / num_rings_each_zone[0],
                         (min_range_z3_ - min_range_z2_) / num_rings_each_zone[1],
                         (min_range_z4_ - min_range_z3_) / num_rings_each_zone[2],
                         (max_range_ - min_range_z4_) / num_rings_each_zone[3]]

sector_sizes_ = [2 * math.pi / num_sectors_each_zone[0], 
                 2 * math.pi / num_sectors_each_zone[1], 
                 2 * math.pi / num_sectors_each_zone[2], 
                 2 * math.pi / num_sectors_each_zone[3]]

### CLUSTERING ###

N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS)

UPRIGHT_ENOUGH  = 0.55 # cyan
FLAT_ENOUGH = 0.2  # green
TOO_HIGH_ELEVATION = 0.0 # blue
TOO_TILTED = 1. # red

# Define the color for each layer
layer_colors = [(255, 0, 0), (255, 165, 0), (0, 255, 0), (0, 0, 255)] # Red, Orange, Green, Blue
cluster_colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)] # Red, Green, Blue
cluster_likelihoods = [0.55, 0.2, 1.0] 

fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
          pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
          pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
          # Channel format: name, offset (in bytes), datatype, count
          pc2.PointField('rgb', 12, pc2.PointField.FLOAT32, 1)]


### DEFINE NODES ###


pub = rospy.Publisher('output', PointCloud2, queue_size=1)
poly_pub = rospy.Publisher('polygons', PolygonArray, queue_size=1)


### CLASS FOR CZM ###


class Zone:
    def __init__(self):
        self.points = []

def xy2radius(x, y):
    return np.sqrt(x**2 + y**2)

def xy2theta(x, y):
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2 * math.pi  # theta가 음수일 경우, 2*pi를 더하여 양수로 변환
    return theta


def rgb_to_float(color):
    """ Converts 8-bit RGB values to a single floating-point number. """
    hex_color = 0x0000 | (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
    return struct.unpack('f', struct.pack('I', hex_color))[0]

def get_likelihood_for_label(label):
    # According to the label value, return the corresponding likelihood
    if label == 0:
        return UPRIGHT_ENOUGH  # cyan
    elif label == 1:
        return FLAT_ENOUGH  # green
    elif label == 2:
        return TOO_HIGH_ELEVATION  # blue
    elif label == 3:
        return TOO_TILTED  # red
    else:
        return 0.0  # for safety, in case an unexpected label value appears


def set_polygons(cloud_msg, zone_idx, r_idx, theta_idx, num_split, ring_size, sector_size, min_range, color):
    # assert len(cluster_colors) == len(cluster_likelihoods), "Number of colors must match number of likelihoods"
    polygon_stamped = PolygonStamped()
    polygon_stamped.header = cloud_msg.header

    # Set point of polygon. Start from RL and ccw
    MARKER_Z_VALUE = 0.0 # You can modify this value

    # RL
    zone_min_range = min_range[zone_idx]
    r_len = r_idx * ring_size[zone_idx] + zone_min_range
    angle = theta_idx * sector_size[zone_idx]

    point = Point32(x = r_len * cos(angle),
                    y = r_len * sin(angle),
                    z = MARKER_Z_VALUE)
    polygon_stamped.polygon.points.append(point)

    # RU
    r_len = r_len + ring_size[zone_idx]
    point = Point32(x = r_len * cos(angle),
                    y = r_len * sin(angle),
                    z = MARKER_Z_VALUE)
    polygon_stamped.polygon.points.append(point)

    # RU -> LU
    for idx in range(1, num_split + 1):
        angle = angle + sector_size[zone_idx] / num_split
        point = Point32(x = r_len * cos(angle),
                        y = r_len * sin(angle),
                        z = MARKER_Z_VALUE)
        polygon_stamped.polygon.points.append(point)

    r_len = r_len - ring_size[zone_idx]
    point = Point32(x = r_len * cos(angle),
                    y = r_len * sin(angle),
                    z = MARKER_Z_VALUE)
    polygon_stamped.polygon.points.append(point)

    for idx in range(1, num_split):
        angle = angle - sector_size[zone_idx] / num_split
        point = Point32(x = r_len * cos(angle),
                        y = r_len * sin(angle),
                        z = MARKER_Z_VALUE)
        polygon_stamped.polygon.points.append(point)

    return polygon_stamped, std_msgs.msg.ColorRGBA(color[0], color[1], color[2], color[3])

all_image_vectors = []

def cloud_cb(cloud_msg):

    cloud = pc2.read_points(cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    # czm = [Zone() for _ in range(4*16)] # for 4 rings and 16 sectors
    # czm = [[[Zone() for _ in range(sector_const)] for _ in range(4)] for _ in range(4)]
    czm = [[
        [Zone() for _ in range(num_sectors_each_zone[zone_idx])]
        for _ in range(num_rings_each_zone[zone_idx])
    ] for zone_idx in range(num_zones)]
    # czm = np.random.rand(num_zones, num_rings, num_sectors, 5)


    for pt in cloud:
        x, y, z, intensity = pt
        r = xy2radius(x, y)
        if r <= max_range_ and r > min_range_:
            theta = xy2theta(x, y)
            ring_idx = sector_idx = 0

            if r < min_range_z2_:
                ring_idx   = min(int((r - min_range_) / ring_sizes_[0]), num_rings_each_zone[0]-1)
                sector_idx = min(int(theta / sector_sizes_[0]), num_sectors_each_zone[0]-1)
                czm[0][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[0]])
            elif r < min_range_z3_:
                ring_idx   = min(int((r - min_range_z2_) / ring_sizes_[1]), num_rings_each_zone[1]-1)
                sector_idx = min(int(theta / sector_sizes_[1]), num_sectors_each_zone[1]-1)
                czm[1][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[1]])
            elif r < min_range_z4_:
                ring_idx   = min(int((r - min_range_z3_) / ring_sizes_[2]), num_rings_each_zone[2]-1)
                sector_idx = min(int(theta / sector_sizes_[2]), num_sectors_each_zone[2]-1)
                czm[2][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[2]])
            else:
                ring_idx   = min(int((r - min_range_z4_) / ring_sizes_[3]), num_rings_each_zone[3]-1)
                sector_idx = min(int(theta / sector_sizes_[3]), num_sectors_each_zone[3]-1)
                czm[3][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[3]])

    all_points = []
    for layer in czm:
        for ring in layer:
            for sector in ring:
                if len(sector.points) > MIN_POINTS:
                    all_points.extend(sector.points)


    z_values = [pt[2] for pt in all_points]
    min_z = min(z_values)
    max_z = max(z_values)

    # Prepare PolygonArray message
    polygon_msg = PolygonArray()
    polygon_msg.header = cloud_msg.header

    colors = []

    img_dim = 15 # The dimension (in pixels) of your output images
    for zone_idx, layer in enumerate(czm):
        for ring_idx, ring in enumerate(layer):
            for sector_idx, sector in enumerate(ring):
                if len(sector.points) > MIN_POINTS:

                    #heightmap
                    image = np.zeros((img_dim, img_dim), dtype=np.uint8)

                    # Normalize x, y coordinates to fit in the image dimensions
                    x_values = [pt[0] for pt in sector.points]
                    y_values = [pt[1] for pt in sector.points]
                    
                    min_x, max_x = min(x_values), max(x_values)
                    min_y, max_y = min(y_values), max(y_values)

                    for pt in sector.points:
                        normalized_x = int((pt[0] - min_x) / (max_x - min_x) * (img_dim - 1))
                        normalized_y = int((pt[1] - min_y) / (max_y - min_y) * (img_dim - 1))
                        normalized_z = int((pt[2] - min_z) / (max_z - min_z) * 255)

                        image[normalized_y, normalized_x] = normalized_z

                    # Store the image vector
                    image_vector = image.reshape(-1, 1)
                    all_image_vectors.append(image_vector)

                    # cv2.imwrite('heightmaps_30/layer{}_ring{}_sector{}.png'.format(zone_idx, ring_idx, sector_idx), image)

    # Once all images have been processed, perform clustering
    scaler = StandardScaler()
    all_image_vectors_scaled = scaler.fit_transform(np.concatenate(all_image_vectors))

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(all_image_vectors_scaled)
    all_labels = kmeans.labels_
    print(all_labels)

    # Iterate over the zones, rings, and sectors again to assign the labels
    label_idx = 0
    for zone_idx, layer in enumerate(czm):
        for ring_idx, ring in enumerate(layer):
            for sector_idx, sector in enumerate(ring):
                if len(sector.points) > MIN_POINTS:
                    cluster_label = all_labels[label_idx]

                    #polygon 
                    polygon_stamped, color = set_polygons(cloud_msg, zone_idx, ring_idx, sector_idx, 10, ring_sizes_, sector_sizes_, min_ranges_, color=cluster_colors[cluster_label])
                    polygon_msg.polygons.append(polygon_stamped)
                    colors.append(color)

                    # Add likelihood to the polygon label
                    polygon_msg.likelihood.append(get_likelihood_for_label(cluster_label))

                    sector.points = [[pt[0], pt[1], pt[2], pt[3], cluster_colors[cluster_label]] for pt in sector.points]
                    label_idx += 1
    poly_pub.publish(polygon_msg)

    # print("Zone 1, Ring 1, Sector Indices: ", sector_indices)  # 각 호출에서의 sector_indices 리스트 출력

    if all_points:
        output = pc2.create_cloud(cloud_msg.header, fields, [(pt[0], pt[1], pt[2], rgb_to_float(pt[4])) for pt in all_points])
        pub.publish(output)


def listener():
    rospy.init_node('bumpypatch_node', anonymous=True)
    rospy.Subscriber("/os1_cloud_node/points", PointCloud2, cloud_cb)
    rospy.spin()

if __name__ == '__main__':
    listener()
