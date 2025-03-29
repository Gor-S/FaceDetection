import torch
from scripts import detect_faces, tools_FF, tools
 
if __name__ == "__main__":
    config = tools.load_config("FF-config.yaml") 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # 1. Extract faces from the video using your VideoProcessor
        print("🔍 Extracting faces from video...")
        video_processor = detect_faces.VideoProcessor(
            video_path=config["video_path"],
            model_path=config["model_path"],
            frame_skip=config["frame_skip"],
            output_dir=config["output_dir"],
            device=device
        )
        video_processor.process()
        
        fps = tools_FF.get_video_fps(config["video_path"])

        # 2. Compute the embedding for the target face
        print("💡 Computing embedding for the target face...")
        target_embedding = tools_FF.compute_target_embedding(config["target_face_path"], device=device)

        # 3. Process extracted faces and search for matches
        print("🔎 Analyzing extracted faces...")
        intervals = tools_FF.process_extracted_faces(
            faces_dir=config["output_dir"],
            target_embedding=target_embedding,
            fps=fps,
            threshold=config["threshold"],
            min_gap_sec=config["min_gap"],
            device=device
        )
        tools.remove_folder(config["output_dir"])
        if intervals:
            for start, end in intervals:
                print(f"Face detected from {start:.2f} sec to {end:.2f} sec")
        else:
            print("Target face not found in the video.")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        tools.remove_folder(config["output_dir"])

