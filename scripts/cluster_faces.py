import os
import shutil
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import hdbscan
from . import logger
from . import progress_bar

get_logger = logger.get_logger

class FaceClustering:
    def __init__(self, faces_folder, output_folder):
        self.faces_folder = faces_folder
        self.output_folder = output_folder
        self.image_extensions = ('.jpg', '.jpeg', '.png')
        os.makedirs(self.output_folder, exist_ok=True)
        self.logger = get_logger("face_clustering")

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
        
        with progress_bar.ProgressBar(total=len(image_paths), desc="Computing embeddings", unit="images") as pbar:
            for image_path in image_paths:
                try:
                    img = Image.open(image_path).convert('RGB')
                    # Filtering by aspect ratio: faces usually have near-square dimensions
                    width, height = img.size
                    aspect_ratio = width / height
                    if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                        self.logger.info(f"Skipping {image_path} due to unusual aspect ratio: {aspect_ratio:.2f}")
                        pbar.update()
                        continue
                    img_tensor = self.transform(img).unsqueeze(0)
                    with torch.no_grad():
                        embedding = self.model(img_tensor)
                    embeddings.append(embedding.squeeze().cpu().numpy())
                    filenames.append(image_path)
                except Exception as e:
                    self.logger.error(f"Error with {image_path}: {e}")
                pbar.update()
        return np.array(embeddings) if embeddings else None, filenames

    def cluster_faces(self):
        image_paths = self.get_image_paths()
        self.logger.info(f"Found {len(image_paths)} images to process")
        
        embeddings, filenames = self.compute_embeddings(image_paths)

        if embeddings is not None:
            # L2-normalize embeddings
            self.logger.info("Normalizing embeddings...")
            embeddings = np.array(embeddings)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-10)

            self.logger.info("Clustering faces...")
            clustering_model = hdbscan.HDBSCAN(min_cluster_size=4, metric='euclidean')
            
            with progress_bar.ProgressBar(total=1, desc="Clustering", unit="batches") as pbar:
                labels = clustering_model.fit_predict(embeddings)
                pbar.update()
                
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            self.logger.info(f"Created {num_clusters} clusters.")

            clusters = {}
            for label, file in zip(labels, filenames):
                clusters.setdefault(label, []).append(file)

            # Move files to cluster folders with progress bar
            total_files = sum(len(files) for label, files in clusters.items() if label != -1)
            with progress_bar.ProgressBar(total=total_files, desc="Moving files to clusters", unit="files") as pbar:
                for cluster_id, files in clusters.items():
                    if cluster_id == -1:
                        continue
                    else:
                        folder_name = os.path.join(self.output_folder, f"person_{cluster_id}")
                    os.makedirs(folder_name, exist_ok=True)
                    for file in files:
                        shutil.move(file, os.path.join(folder_name, os.path.basename(file)))
                        pbar.update()

            self.logger.info("Sorting completed.")
        else:
            self.logger.error("Failed to compute embeddings for any image.")