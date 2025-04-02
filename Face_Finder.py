import torch
from scripts import detector, tools_FF, tools, compatibility, banner
import logging


apply_all_patches = compatibility.apply_all_patches

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/face_finder.log")
    ]
)
logger = logging.getLogger("FaceFinder")

if __name__ == "__main__":
    banner.print_banner_FE()
    # Apply compatibility patches for deprecation warnings
    apply_all_patches()
    
    config = tools.load_config("config/config.yaml")
    ff_config = config.get("face_finder", {})
    video_path = input("Enter the path to the video: ").strip()
    target_face_path = input("Enter the path to the image: ").strip()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        print("🔍 Extracting faces from video...")
        video_processor = detector.VideoProcessor(
            video_path,
            model_path=ff_config.get("model_path"),
            frame_skip=ff_config.get("frame_skip", 5),
            output_dir=ff_config.get("output_dir", "FF_output_dir"),
            device=device
        )
        video_processor.process()
        
        fps = tools_FF.get_video_fps(video_path)

        print("💡 Computing embedding for the target face...")
        target_embedding = tools_FF.compute_target_embedding(target_face_path, device=device)

        print("🔎 Analyzing extracted faces...")
        intervals = tools_FF.process_extracted_faces(
            faces_dir=ff_config.get("output_dir", "FF_output_dir"),
            target_embedding=target_embedding,
            fps=fps,
            threshold=ff_config.get("threshold", 0.6),
            min_gap_sec=ff_config.get("min_gap", 1.0),
            device=device
        )

        if not ff_config.get("keep_intermediate_files", False):
            tools.remove_folder(ff_config.get("output_dir", "FF_output_dir"))
            
        if intervals:
            for start, end in intervals:
                print(f"Face detected from {start:.2f} sec to {end:.2f} sec")

            if ff_config.get("output_timecodes", False):
                with open("detection_timecodes.txt", "w") as f:
                    for start, end in intervals:
                        start_tc = tools_FF.seconds_to_timecode(start, fps)
                        end_tc = tools_FF.seconds_to_timecode(end, fps)
                        f.write(f"{start_tc} --> {end_tc}\n")
                logger.info("Saved timecodes to detection_timecodes.txt")
        else:
            print("Target face not found in the video.")

    except Exception as e:
        logger.exception(f"❌ Error during processing: {e}")
        tools.remove_folder(ff_config.get("output_dir", "FF_output_dir"))