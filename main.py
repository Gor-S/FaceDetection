import gc
import torch
import yaml
from scripts import detect_faces, grouping_of_persons, choosing_the_best_frame, photo_enhancement

# Load config file
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("❌ Error: Config file not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")
        exit(1)

config = load_config()

def clear_memory():
    """ Free memory and clear GPU cache if available. """
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠️ GPU memory cleanup error: {e}")

if __name__ == "__main__":
    try:
        # 1. Face detection
        print("🔍 Detecting faces...")
        processor = detect_faces.VideoProcessor(
            config["video_path"], 
            config["face_model_path"],
            config["frame_skip"],
            config["faces_output_dir"]
        )
        processor.process()
        clear_memory()

        # 2. Face clustering
        print("📂 Clustering faces...")
        clusterer = grouping_of_persons.FaceClustering(
            config["faces_output_dir"], 
            config["clusters_output_dir"]
        )
        clusterer.cluster_faces()
        clear_memory()

        # 3. Selecting the best frames
        print("📸 Selecting best frames...")
        choosing_the_best_frame.FolderProcessor.process_all_folders(
            config["clusters_output_dir"], 
            config["best_faces_dir"]
        )

        # Cleanup temporary clustered faces
        choosing_the_best_frame.clear_clustered_faces(config["clusters_output_dir"])
        choosing_the_best_frame.clear_clustered_faces(config["faces_output_dir"])
        clear_memory()

        # 4. Photo enhancement
        print("✨ Enhancing faces...")
        photo_enhancement.process_folders(
            config["best_faces_dir"],
            config["GFPGAN_model_path"]
        )

        print("✅ Processing complete!")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
