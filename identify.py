import os
import cv2
import shutil
import face_recognition
from sklearn.cluster import DBSCAN
import numpy as np
from tqdm import tqdm
import pickle

# Step 1: Face Detection and Encoding
faces_folder = "faces_detected"
image_paths = [os.path.join(faces_folder, f) for f in os.listdir(faces_folder)]

# Load known encodings from the file
encodings_file = "known_encodings.pkl"

try:
    with open(encodings_file, 'rb') as f:
        known_encodings, image_names = pickle.load(f)
        # Convert to lists if they are not already lists
        known_encodings = known_encodings.tolist() if isinstance(known_encodings, np.ndarray) else known_encodings
        image_names = image_names.tolist() if isinstance(image_names, np.ndarray) else image_names
    print("Known encodings loaded from:", encodings_file)
except FileNotFoundError:
    print("No previous known encodings found.")
    known_encodings = []
    image_names = []

# Remaining steps...
# Track processed filenames
processed_files = set(image_names)

# Append new encodings to the existing list
for image_path in tqdm(image_paths, desc="Face Detection and Encoding"):
    # Check if the image has already been processed
    if os.path.basename(image_path) in processed_files:
        continue

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    if len(face_encodings) > 0:
        known_encodings.append(face_encodings[0])  # Take only the first face encoding
        image_names.append(os.path.basename(image_path))
        # Add the processed filename to the set
        processed_files.add(os.path.basename(image_path))

known_encodings = np.array(known_encodings)
# Save updated known encodings back to the file
with open(encodings_file, 'wb') as f:
    pickle.dump((known_encodings, np.array(image_names)), f)

print("Known encodings updated and saved to:", encodings_file)

# Step 2: Face Comparison
similar_faces = []
for i in range(len(known_encodings)):
    for j in range(i + 1, len(known_encodings)):
        distance = face_recognition.face_distance([known_encodings[i]], known_encodings[j])
        if distance < 0.9:  # Adjust threshold as needed
            similar_faces.append((i, j))

# Step 3: Clustering
clusters = DBSCAN(eps=0.5, min_samples=2).fit_predict(known_encodings)

# Step 4: Combine Faces
combined_faces = {}
for cluster_id in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster_id)[0]
    if len(cluster_indices) > 1:  # Only combine if there are multiple faces in the cluster
        combined_face = known_encodings[cluster_indices].mean(axis=0)
        combined_faces[cluster_id] = combined_face

# Step 5: Save Clustered Faces
output_folder = "clustered_faces"
os.makedirs(output_folder, exist_ok=True)

for cluster_id, face_encoding in combined_faces.items():
    cluster_folder = os.path.join(output_folder, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)
    for idx in np.where(clusters == cluster_id)[0]:
        image_name = image_names[idx]
        original_image_path = os.path.join(faces_folder, image_name)
        new_image_path = os.path.join(cluster_folder, image_name)
        shutil.copyfile(original_image_path, new_image_path)  # Make a copy of the image

print("Clustered faces saved in the 'clustered_faces' directory.")
