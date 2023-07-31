import os
import shutil
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from PIL import Image

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')
        if img is not None:
            img = np.array(img).flatten()  # Convert the image to a 1D array
            images.append(img)
            filenames.append(filename)
    return images, filenames

folder = '/home/rtlink/jiwon/paper_ws/heightmaps'  # replace with your folder path
images, filenames = load_images_from_folder(folder)

# Scale data
scaler = StandardScaler()
images = scaler.fit_transform(images)

# DBSCAN
db = DBSCAN(eps=0.3, min_samples=5).fit(images)

labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# Create a new directory for each cluster and copy the images into the correct directories
for i in range(len(labels)):
    label = labels[i]
    filename = filenames[i]
    directory = os.path.join(folder, str(label))
    if not os.path.exists(directory):
        os.makedirs(directory)
    shutil.copy2(os.path.join(folder, filename), directory)
