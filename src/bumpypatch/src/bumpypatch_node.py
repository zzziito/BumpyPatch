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
from sklearn.decomposition import PCA
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler
import std_msgs
import tf

### DEFINE CONSTANTS ###


MIN_POINTS = 30

min_range_ = 4
min_range_z2_ = 12
min_range_z3_ = 28
min_range_z4_ = 60
max_range_ = 100.0

min_ranges_   = [min_range_, min_range_z2_, min_range_z3_, min_range_z4_]

sector_const = 32

num_zones = 4
num_sectors_each_zone = [36, 48 ,48, 32]
num_rings_each_zone = [4, 4, 4, 5]

ring_sizes_   = [(min_range_z2_ - min_range_) / num_rings_each_zone[0],
                         (min_range_z3_ - min_range_z2_) / num_rings_each_zone[1],
                         (min_range_z4_ - min_range_z3_) / num_rings_each_zone[2],
                         (max_range_ - min_range_z4_) / num_rings_each_zone[3]]

sector_sizes_ = [2 * math.pi / num_sectors_each_zone[0], 
                 2 * math.pi / num_sectors_each_zone[1], 
                 2 * math.pi / num_sectors_each_zone[2], 
                 2 * math.pi / num_sectors_each_zone[3]]

angle_threshold = 2

### CLUSTERING ###

N_CLUSTERS = 3
kmeans = KMeans(n_clusters=N_CLUSTERS)

UPRIGHT_ENOUGH  = 0.55 # cyan
FLAT_ENOUGH = 0.2  # green
TOO_HIGH_ELEVATION = 0.0 # blue
TOO_TILTED = 1. # red

# Define the color for each layer
layer_colors = [(255, 0, 0), (255, 165, 0), (0, 255, 0), (0, 0, 255)] # Red, Orange, Green, Blue

# cluster_colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)  ] # Red, Green, Blue
# cluster_likelihoods = [0.55, 0.2, 1.0, 0.0] 

cluster_colors = [(0.0, 0.0, 1.0, 1.0), (0.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0)] # Blue, Green, Red
cluster_likelihoods = [0.55, 0.2, 1.0] 

fields = [pc2.PointField('x', 0, pc2.PointField.FLOAT32, 1),
          pc2.PointField('y', 4, pc2.PointField.FLOAT32, 1),
          pc2.PointField('z', 8, pc2.PointField.FLOAT32, 1),
          # Channel format: name, offset (in bytes), datatype, count
          pc2.PointField('rgb', 12, pc2.PointField.FLOAT32, 1)]


### DEFINE NODES ###


pub = rospy.Publisher('output', PointCloud2, queue_size=1)
poly_pub = rospy.Publisher('polygons', PolygonArray, queue_size=1)
marker_pub = rospy.Publisher('normal_vector_marker', Marker, queue_size=10)

traversable_pub = rospy.Publisher("traversable_sector", PointCloud2, queue_size=10)
traversable_poly_pub = rospy.Publisher("traversable_sector_poly", PolygonArray, queue_size=10)
traversable_poly_marker_pub = rospy.Publisher('traversable_sector_marker', Marker, queue_size=10)



### CLASS FOR CZM ###


class Zone:
    def __init__(self):
        self.points = []

### czm 을 위한 함수

def xy2radius(x, y):
    return np.sqrt(x**2 + y**2)

def xy2theta(x, y):
    theta = math.atan2(y, x)
    if theta < 0:
        theta += 2 * math.pi  # theta가 음수일 경우, 2*pi를 더하여 양수로 변환
    return theta


def rgb_to_float(color):
    hex_color = 0x0000 | (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
    return struct.unpack('f', struct.pack('I', hex_color))[0]

### jsk_msgs polygon 의 likelihood 에 따라 색상 visualize

# def get_likelihood_for_label(label):
#     # According to the label value, return the corresponding likelihood
#     if label == 0:
#         return UPRIGHT_ENOUGH  # cyan
#     elif label == 1:
#         return FLAT_ENOUGH  # green
#     elif label == 2:
#         return TOO_HIGH_ELEVATION  # blue
#     elif label == 3:
#         return TOO_TILTED  # red
#     else:
#         return 0.0  # for safety, in case an unexpected label value appears

def get_likelihood_for_label(label):
    # According to the label value, return the corresponding likelihood
    if label == 0:
        return UPRIGHT_ENOUGH 
    elif label == 1:
        return TOO_HIGH_ELEVATION  
    elif label == 2:
        return TOO_TILTED  
    else:
        return 0.0  # for safety, in case an unexpected label value appears



def set_polygons(cloud_msg, zone_idx, r_idx, theta_idx, num_split, ring_size, sector_size, min_range):
    # assert len(cluster_colors) == len(cluster_likelihoods), "Number of colors must match number of likelihoods"
    polygon_stamped = PolygonStamped()
    polygon_stamped.header = cloud_msg.header

    # Set point of polygon. Start from RL and ccw
    MARKER_Z_VALUE = -2.0 # You can modify this value

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

    return polygon_stamped

all_image_vectors = []

### sector 별 normal vector

def calculate_normal_vector(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    normal_vector = pca.components_[-1]  # The normal vector is the eigenvector with the smallest eigenvalue

    normal_vector = np.array(normal_vector)
    angle_with_x_axis = np.arccos(np.dot(normal_vector, [1, 0, 0]))
    angle_with_y_axis = np.arccos(np.dot(normal_vector, [0, 1, 0]))
    angle_with_z_axis = np.arccos(np.dot(normal_vector, [0, 0, 1]))
    normal_vector = [angle_with_x_axis, angle_with_y_axis, angle_with_z_axis]
    # print(normal_vector)
    return normal_vector

def normal_vector_to_msg(normal_vector, x, y, z, marker_id):
    x = -x
    y = -y

    marker = Marker()
    marker.header.frame_id = "base_link"  # Replace with the appropriate frame ID
    marker.header.stamp = rospy.Time.now()
    marker.ns = "normal_vectors"
    marker.id = marker_id  # Unique ID for each marker
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # Set marker position to the center of the sector
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 1

    # Calculate the magnitude of the normal vector
    magnitude = np.linalg.norm(normal_vector)

    # Set the marker scale
    marker.scale.x = magnitude  # Length of the arrow
    marker.scale.y = 0.1      # Width of the arrow
    marker.scale.z = 0.1       # Height of the arrow

    quat = quaternion_from_euler(normal_vector[0], normal_vector[1], normal_vector[2])

    # print("quaternion : " , quat)

    # Set the orientation of the marker
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]

    # Set marker color and transparency
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0

    marker.lifetime = rospy.Duration()

    return marker

def average_direction(vectors):
    return np.mean(vectors, axis=0)

def angle_with_z_axis(vector):
    vector = vector / np.linalg.norm(vector)
    return np.arccos(np.dot(vector, [0, 0, -1]))

### image processing ###

def apply_gabor_filter(image, angle):
    ksize = 15
    sigma = 5
    lambd = 3
    gamma = 0.5
    psi = 0  # phase offset
    theta = angle * np.pi / 8

    gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
    
    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_filter)

    # 임계값 설정 및 스트라이프 영역 검출
    threshold = np.max(filtered_image) * 0.9
    stripe_regions = filtered_image > threshold
    stripe_image = np.uint8(stripe_regions * 255)

    return stripe_image
    # return filtered_image

def count_stripes(image, min_length=5):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for contour in contours:
        length = cv2.arcLength(contour, True) # 외곽선의 길이 계산
        if length > min_length: # 길이가 임계값보다 크면 카운트
            count += 1
    return count

def extract_features(image, image_sector_index, images_to_cluster, labels, max_stripes_list):
    max_stripes = 0
    best_image = None

    for angle in range(8):
        filtered_image = apply_gabor_filter(image, angle)
        num_stripes = count_stripes(filtered_image)
        if num_stripes > max_stripes:
            max_stripes = num_stripes
            best_image = filtered_image

    if best_image is not None:
        # print(max_stripes)
        
        images_to_cluster.append(best_image.flatten())
        labels.append(image_sector_index)
        max_stripes_list.append(max_stripes)
    else :
        best_image = np.zeros((15, 15), dtype=np.uint8)
        images_to_cluster.append(best_image.flatten())
        labels.append(image_sector_index)
        max_stripes_list.append(max_stripes)

def remove_specific_pattern(image, threshold):
    # 패딩 추가 (상, 하, 좌, 우 1픽셀씩)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

    # 원래 이미지와 패딩된 이미지의 차이 계산 (상하좌우)
    diff_up = padded_image[:-2, 1:-1] - padded_image[1:-1, 1:-1]
    diff_down = padded_image[2:, 1:-1] - padded_image[1:-1, 1:-1]
    diff_left = padded_image[1:-1, :-2] - padded_image[1:-1, 1:-1]
    diff_right = padded_image[1:-1, 2:] - padded_image[1:-1, 1:-1]

    # 조건에 맞는 픽셀 찾기 (상하좌우 차이가 모두 threshold보다 큰 경우)
    condition = (diff_up > threshold) & (diff_down > threshold) & (diff_left > threshold) & (diff_right > threshold)

    # 조건에 맞는 픽셀을 0으로 설정
    result_image = np.copy(image)
    result_image[condition] = 0

    return result_image

def analyze_cluster(labels, cluster_label, max_stripes_list):
    cluster_stripes = [stripe for idx, stripe in enumerate(max_stripes_list) if labels[idx] == cluster_label]
    if cluster_stripes:
        mean_stripe_count = np.mean(cluster_stripes)
    else:
        mean_stripe_count = 0
    return mean_stripe_count

### callback function ###

def cloud_cb(cloud_msg):

    cloud = pc2.read_points(cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    czm = [[
        [Zone() for _ in range(num_sectors_each_zone[zone_idx])]
        for _ in range(num_rings_each_zone[zone_idx])
    ] for zone_idx in range(num_zones)]

    all_points = []
    traversable_points = []
    ring_max_points_counts = []
    marker_id = 0

    images = []
    image_sector_indices = []

    images_to_cluster = []
    labels = []
    max_stripes_list = []

    sector_cluster_mapping = {}
    cluster_stripe_counts = {}

    for pt in cloud:
        x, y, z, intensity = pt
        r = xy2radius(x, y)
        if r <= max_range_ and r > min_range_:
            theta = xy2theta(x, y)
            zone_idx = ring_idx = sector_idx = 0

            if r < min_range_z2_:
                zone_idx = 0
                ring_idx   = min(int((r - min_range_) / ring_sizes_[0]), num_rings_each_zone[0]-1)
                sector_idx = min(int(theta / sector_sizes_[0]), num_sectors_each_zone[0]-1)
                czm[zone_idx][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[0]])
            elif r < min_range_z3_:
                zone_idx = 1
                ring_idx   = min(int((r - min_range_z2_) / ring_sizes_[1]), num_rings_each_zone[1]-1)
                sector_idx = min(int(theta / sector_sizes_[1]), num_sectors_each_zone[1]-1)
                czm[zone_idx][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[1]])
            elif r < min_range_z4_:
                zone_idx = 2
                ring_idx   = min(int((r - min_range_z3_) / ring_sizes_[2]), num_rings_each_zone[2]-1)
                sector_idx = min(int(theta / sector_sizes_[2]), num_sectors_each_zone[2]-1)
                czm[zone_idx][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[2]])
            else:
                zone_idx = 3
                ring_idx   = min(int((r - min_range_z4_) / ring_sizes_[3]), num_rings_each_zone[3]-1)
                sector_idx = min(int(theta / sector_sizes_[3]), num_sectors_each_zone[3]-1)
                czm[zone_idx][ring_idx][sector_idx].points.append([x, y, z, intensity, layer_colors[3]])


    # Prepare PolygonArray message
    polygon_msg = PolygonArray()
    polygon_msg.header = cloud_msg.header

    traversable_polygon_msg = PolygonArray()
    traversable_polygon_msg.header = cloud_msg.header


    img_dim = 15 # The dimension (in pixels) of your output images
    for zone_idx, layer in enumerate(czm):
        for ring_idx, ring in enumerate(layer):
            for sector_idx, sector in enumerate(ring):
                if len(sector.points) > MIN_POINTS:
                    #normal vector  
                    all_points.extend(sector.points)
                    points = np.array([pt[:3] for pt in sector.points])  # Use only the x, y, z coordinates
                    normal_vector = calculate_normal_vector(points)

                    # # If this is the first sector in this cluster, initialize the list of normal vectors
                    # if cluster_label not in cluster_to_avg_direction:
                    #     cluster_to_avg_direction[cluster_label] = []

                    # # Append the normal vector to the list of normal vectors for this cluster
                    # cluster_to_avg_direction[cluster_label].append(normal_vector)


                    # Calculate sector center
                    sector_center_x = np.mean(points[:, 0])
                    sector_center_y = np.mean(points[:, 1])
                    sector_center_z = np.mean(points[:, 2])

                    # Create marker for the normal vector
                    marker = normal_vector_to_msg(normal_vector, sector_center_x, sector_center_y, sector_center_z, marker_id)
                    marker_id += 1  # Increment the marker ID
                    
                    # Publish the Marker
                    marker_pub.publish(marker)
                    # print(angle_with_z_axis(normal_vector))
                    

                    # 3. In the loop where sectors are processed, add the following code:
                    if angle_with_z_axis(normal_vector) >= 1.1 and angle_with_z_axis(normal_vector) <= 1.9:

                        # print(angle_with_z_axis(normal_vector))
                        # This is a traversable sector, publish its points and polygon
                        traversable_points.extend(sector.points)

                        # traversable_polygon_stamped = set_polygons(cloud_msg, zone_idx, ring_idx, sector_idx, 10, ring_sizes_, sector_sizes_, min_ranges_)
                        # traversable_polygon_msg.polygons.append(traversable_polygon_stamped)
                        # traversable_poly_pub.publish(traversable_polygon_msg)
                        # traversable_poly_marker_pub.publish(marker)

                        #heightmap
                        image = np.zeros((img_dim, img_dim), dtype=np.uint8)

                        # Normalize x, y coordinates to fit in the image dimensions
                        x_values = [pt[0] for pt in sector.points]
                        y_values = [pt[1] for pt in sector.points]
                        z_values = [pt[2] for pt in sector.points]
                        
                        min_x, max_x = min(x_values), max(x_values)
                        min_y, max_y = min(y_values), max(y_values)
                        min_z, max_z = min(z_values), max(z_values)

                        for pt in sector.points:
                            normalized_x = int((pt[0] - min_x) / (max_x - min_x) * (img_dim - 1))
                            normalized_y = int((pt[1] - min_y) / (max_y - min_y) * (img_dim - 1))
                            normalized_z = int((pt[2] - min_z) / (max_z - min_z) * 255)

                            image[normalized_y, normalized_x] = normalized_z

                        # cv2.imwrite('heightmaps_30/layer{}_ring{}_sector{}.png'.format(zone_idx, ring_idx, sector_idx), image)
                        image_2d = image.reshape(img_dim, img_dim)
                        image_2d = remove_specific_pattern(image_2d,15)

                        images.append(image_2d)
                        image_sector_indices.append((zone_idx, ring_idx, sector_idx)) # 해당 이미지의 sector 인덱스 추가

                        # Store the image vector
                        image_vector = image_2d.reshape(-1, 1)
                        all_image_vectors.append(image_vector)


    for img, sector_index in zip(images, image_sector_indices):
        extract_features(img, sector_index, images_to_cluster, labels, max_stripes_list)

    kmeans = KMeans(n_clusters=N_CLUSTERS).fit(np.array(max_stripes_list).reshape(-1, 1))
    all_labels = kmeans.labels_
    print(all_labels)

    for i in range(N_CLUSTERS):
        cluster_stripe_counts = {i: analyze_cluster(kmeans.labels_, i, max_stripes_list) for i in range(N_CLUSTERS)}

    # Sort clusters by mean stripe count
    sorted_clusters = sorted(cluster_stripe_counts.items(), key=lambda x: x[1], reverse=True)
    print(sorted_clusters)

    # Mapping according to the sorted_clusters order
    label_mapping = {sorted_clusters[i][0]: i for i in range(len(sorted_clusters))}

    # Apply the mapping to all_labels
    mapped_labels = [label_mapping[label] for label in all_labels]


    # Iterate over the zones, rings, and sectors again to assign the labels
    label_idx = 0
    for zone_idx, layer in enumerate(czm):
        for ring_idx, ring in enumerate(layer):
            for sector_idx, sector in enumerate(ring):
                if len(sector.points) > MIN_POINTS:
                    #normal vector  
                    points = np.array([pt[:3] for pt in sector.points])  # Use only the x, y, z coordinates
                    normal_vector = calculate_normal_vector(points)
                    if angle_with_z_axis(normal_vector) >= 1.1 and angle_with_z_axis(normal_vector) <= 1.9:
                        cluster_label = mapped_labels[label_idx]

                        #polygon 
                        polygon_stamped = set_polygons(cloud_msg, zone_idx, ring_idx, sector_idx, 10, ring_sizes_, sector_sizes_, min_ranges_)
                        polygon_msg.polygons.append(polygon_stamped)

                        # Add likelihood to the polygon label
                        polygon_msg.likelihood.append(get_likelihood_for_label(cluster_label))
                        label_idx += 1
                    
    poly_pub.publish(polygon_msg)
    # traversable_poly_pub.publish(polygon_msg)

    if all_points:
        output = pc2.create_cloud(cloud_msg.header, fields, [(pt[0], pt[1], pt[2], rgb_to_float(pt[4])) for pt in all_points])
        traversable_output = pc2.create_cloud(cloud_msg.header, fields, [(pt[0], pt[1], pt[2], rgb_to_float(pt[4])) for pt in traversable_points])

        pub.publish(output)
        traversable_pub.publish(traversable_output)


def listener():
    rospy.init_node('bumpypatch_node', anonymous=True)
    rospy.Subscriber("/os1_cloud_node/points", PointCloud2, cloud_cb)
    rospy.spin()

if __name__ == '__main__':
    listener()
