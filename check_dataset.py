import os
import cv2

# Path to the dataset
dataset_path = "dataset/data"

# Get all files
files = os.listdir(dataset_path)

# Keep only JPG images
images = [file for file in files if file.lower().endswith(".jpg")]

print("Total JPG images:", len(images))

print("\nFirst 10 images:")

for image in images[:10]:
    print(image)

# Read the first image
if len(images) > 0:
    first_image = os.path.join(dataset_path, images[0])

    img = cv2.imread(first_image)

    if img is not None:
        print("\nFirst image loaded successfully!")
        print("Image name:", images[0])
        print("Image dimensions:", img.shape)
    else:
        print("\nCould not read the image.")