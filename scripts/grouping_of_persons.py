import os
import shutil
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import hdbscan

class FaceClustering:
    def __init__(self, faces_folder, output_folder):
        self.faces_folder = faces_folder
        self.output_folder = output_folder
        self.image_extensions = ('.jpg', '.jpeg', '.png')
        os.makedirs(self.output_folder, exist_ok=True)

        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_image_paths(self):
        return [os.path.join(self.faces_folder, f) for f in os.listdir(self.faces_folder) if f.lower().endswith(self.image_extensions)]

    def compute_embeddings(self, image_paths):
        embeddings, filenames = [], []
        for image_path in image_paths:
            try:
                img = Image.open(image_path).convert('RGB')
                # Filtering by aspect ratio: faces usually have near-square dimensions
                width, height = img.size
                aspect_ratio = width / height
                if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                    print(f"Skipping {image_path} due to unusual aspect ratio: {aspect_ratio:.2f}")
                    continue
                img_tensor = self.transform(img).unsqueeze(0)
                with torch.no_grad():
                    embedding = self.model(img_tensor)
                embeddings.append(embedding.squeeze().cpu().numpy())
                filenames.append(image_path)
            except Exception as e:
                print(f"Error with {image_path}: {e}")
        return np.array(embeddings) if embeddings else None, filenames

    def cluster_faces(self):
        image_paths = self.get_image_paths()
        embeddings, filenames = self.compute_embeddings(image_paths)

        if embeddings is not None:
            # L2-normalize embeddings
            embeddings = np.array(embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

            # Apply HDBSCAN with min_cluster_size=4 and metric 'euclidean'
            # For normalized vectors, Euclidean distance is equivalent to cosine distance.
            clustering_model = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
            labels = clustering_model.fit_predict(embeddings)
            print(f"Created {len(set(labels))-1} clusters.")

            clusters = {}
            for label, file in zip(labels, filenames):
                clusters.setdefault(label, []).append(file)

            for cluster_id, files in clusters.items():

                if cluster_id == -1:
                    continue

                else:
                    folder_name = os.path.join(self.output_folder, f"person_{cluster_id}")
                os.makedirs(folder_name, exist_ok=True)
                for file in files:
                    shutil.move(file, os.path.join(folder_name, os.path.basename(file)))

            print("Sorting completed.")
        else:
            print("Failed to compute embeddings for any image.")