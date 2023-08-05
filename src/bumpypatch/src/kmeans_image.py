import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
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

folder = '/home/rtlink/jiwon/paper_ws/heightmaps_30'  # replace with your folder path
images, filenames = load_images_from_folder(folder)

# Scale data
scaler = StandardScaler()
images = scaler.fit_transform(images)

n_clusters = 4  # Set the number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="random").fit(images)

labels = kmeans.labels_

# Create a new directory for each cluster and copy the images into the correct directories
for i in range(len(labels)):
    label = labels[i]
    filename = filenames[i]
    directory = os.path.join(folder, str(label))
    if not os.path.exists(directory):
        os.makedirs(directory)
    shutil.copy2(os.path.join(folder, filename), directory)
