import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
    accuracy_score
)

# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

dataset_path = "dataset/data"
labels_file = "labels.csv"


# ---------------------------------------------------------
# 2. Load labels
# ---------------------------------------------------------

labels_df = pd.read_csv(labels_file)

print("Total labeled images:", len(labels_df))

print("\nClass distribution:")
print(labels_df["label"].value_counts())


# ---------------------------------------------------------
# 3. Image preprocessing and feature extraction
# ---------------------------------------------------------

X = []
y = []

IMAGE_SIZE = (64, 64)

for _, row in labels_df.iterrows():

    image_name = row["image"]
    label = row["label"]

    image_path = os.path.join(dataset_path, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read:", image_name)
        continue

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image
    resized = cv2.resize(gray, IMAGE_SIZE)

    # Normalize pixel values
    normalized = resized / 255.0

    # Flatten image into feature vector
    features = normalized.flatten()

    X.append(features)
    y.append(label)


X = np.array(X)
y = np.array(y)

print("\nFeature extraction completed.")
print("Feature matrix shape:", X.shape)
print("Label array shape:", y.shape)


# ---------------------------------------------------------
# 4. Split dataset into training and testing sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# ---------------------------------------------------------
# 5. Train Support Vector Machine
# ---------------------------------------------------------

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

print("\nSVM model training completed.")


# ---------------------------------------------------------
# 6. Make predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------
# 7. Calculate Performance
# ---------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))


# ---------------------------------------------------------
# 8. Classification Report
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# 9. Confusion Matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Style A", "Style B"]
)

disp.plot()

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 10. ROC Curve
# ---------------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

roc_auc = auc(
    fpr,
    tpr
)

print("ROC-AUC:", round(roc_auc, 4))


plt.figure()

plt.plot(
    fpr,
    tpr,
    label="SVM (AUC = {:.2f})".format(roc_auc)
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 11. Completion Message
# ---------------------------------------------------------

print("\n================================")
print("Program completed successfully!")
print("================================")

print("\nSaved files:")
print("1. confusion_matrix.png")
print("2. roc_curve.png")