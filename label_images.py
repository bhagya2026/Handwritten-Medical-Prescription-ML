import os
import cv2
import csv

dataset_path = "dataset/data"
labels_file = "labels.csv"

# Read already labeled images
labeled_images = set()

if os.path.exists(labels_file):
    with open(labels_file, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            labeled_images.add(row["image"])

# Get all images
images = sorted(
    [file for file in os.listdir(dataset_path)
     if file.lower().endswith(".jpg")],
    key=lambda x: int(os.path.splitext(x)[0])
)

# Open labels file in append mode
with open(labels_file, "a", newline="") as file:
    writer = csv.writer(file)

    for image_name in images:

        # Skip images that are already labeled
        if image_name in labeled_images:
            continue

        image_path = os.path.join(dataset_path, image_name)
        img = cv2.imread(image_path)

        if img is None:
            continue

        cv2.imshow(
            "Prescription Image - 0: Style A | 1: Style B",
            img
        )

        print(f"\nCurrent image: {image_name}")
        print("0 = Style A (neat/clear)")
        print("1 = Style B (less neat/irregular)")
        print("Q = Quit")

        while True:
            choice = input("Enter 0 or 1: ").strip().lower()

            if choice in ["0", "1"]:
                writer.writerow([image_name, int(choice)])
                file.flush()
                break

            elif choice == "q":
                cv2.destroyAllWindows()
                print("\nLabeling stopped.")
                print("Your previous labels are safely")
                print("Labels saved in:", labels_file)
                exit()

            else:
                print("Please enter only 0, 1, or Q.")

cv2.destroyAllWindows()

print("\nAll 129 images have been labeled!")
print("Labels saved in:", labels_file)