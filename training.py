import cv2
import numpy as np
from PIL import Image
import os

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Path to the dataset
path = "datasets"

def getImageID(path):
    if not os.path.exists(path):
        print(f"Error: Dataset path '{path}' not found.")
        return [], []

    imagePath = [os.path.join(path, f) for f in os.listdir(path)]
    if not imagePath:
        print(f"Error: No images found in '{path}'.")
        return [], []

    faces = []
    ids = []

    for imagePaths in imagePath:
        try:
            faceImage = Image.open(imagePaths).convert('L')
            faceNP = np.array(faceImage)
            Id = int(os.path.split(imagePaths)[-1].split(".")[1])
            faces.append(faceNP)
            ids.append(Id)
        except Exception as e:
            print(f"Error loading image '{imagePaths}': {e}")

    return ids, faces

# Get training data
IDs, facedata = getImageID(path)

# Train the recognizer
recognizer.train(facedata, np.array(IDs))

# Save the trained model
recognizer.write("Trainer.yml")

print("Training complete... Model saved successfully.")

