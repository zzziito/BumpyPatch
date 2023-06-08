import argparse
import os
import pandas as pd
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn
from torchvision.models import resnet50
import numpy as np

def convert_label(label):
    if label == 'blurred_noise_sin1_cropped2':
        return 'noise_sin'
    elif label == 'flat_cropped2':
        return 'flat'
    elif label == 'high_noise_cropped2':
        return 'high_noise'
    elif label == 'low_noise_cropped2':
        return 'low_noise'
    else:
        return label

def main(image_folder, output_csv):
    patch_info = pd.DataFrame(columns=['image', 'label'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("GPU is available")
    else:
        print("GPU is not available, using CPU instead")
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root='/home/rtlink/jiwon/socap_ws/src/heightmap/heightmap_final/final/blurred_copy/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_train_images = len(train_dataset)
    num_classes = len(train_dataset.classes)

    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load('./resnet50_10_26136images.pth'))

    model.eval()

    for i, image_name in enumerate(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('L')
        image_np = np.array(image)

        zero_pixel_ratio = np.sum(image_np == 0) / image_np.size

        if zero_pixel_ratio < 0.3:
            image_tensor = transform(image).unsqueeze(0).to(device)
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_label = train_dataset.classes[predicted]
        else:
            predicted_label = 'none'
        
        predicted_label = convert_label(predicted_label)
        patch_info = patch_info.append({'image': image_name, 'label': predicted_label}, ignore_index=True)

    patch_info.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_folder', type=str, help='Path to image folder')
    parser.add_argument('output_csv', type=str, help='Path to output csv file')

    args = parser.parse_args()
    main(args.image_folder, args.output_csv)
