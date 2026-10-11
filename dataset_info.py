import os

dataset_path = "dataset/data"

images = [
    file for file in os.listdir(dataset_path)
    if file.lower().endswith(".jpg")
]

print("Total prescription images:", len(images))

print("\nImage filenames:")
print(images)