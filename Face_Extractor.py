from scripts import detector, cluster_faces, best_frame, enhancer, PEBC, tools, compatibility, banner
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning)   

apply_all_patches = compatibility.apply_all_patches

if __name__ == "__main__":
    
    try:
        banner.print_banner_FE()
        # Apply compatibility patches for deprecation warnings
        apply_all_patches()
        
        config = tools.load_config("config/config.yaml")
        fe_config = config.get("face_extractor", {})

        print("Enter the path to the video: ")
        video_path = input().strip()
        os.makedirs(fe_config.get("faces_output_dir", "./output/faces"), exist_ok=True)
        os.makedirs(fe_config.get("best_faces_dir", "./output/best_faces"), exist_ok=True)
        
        # 1. Face detection
        print("🔍 Detecting faces...")
        processor = detector.VideoProcessor(
            video_path, 
            fe_config.get("face_model_path"),
            fe_config.get("frame_skip", 5),
            fe_config.get("faces_output_dir")
        )
        processor.process()
        tools.clear_memory()
    
        # 2. Face clustering
        PEBC.process_images_in_folder(fe_config.get("faces_output_dir"))
        print("📂 Clustering faces...")
        clusterer = cluster_faces.FaceClustering(
            fe_config.get("faces_output_dir"), 
            fe_config.get("clusters_output_dir")
        )
        clusterer.cluster_faces()
        tools.clear_memory()
        
        # 3. Selecting the best frames
        best_faces_dir = tools.generate_folder_name()
        print("📸 Selecting best frames...")
        best_frame.FolderProcessor.process_all_folders(
            fe_config.get("clusters_output_dir"), 
            best_faces_dir
        )

        # Cleanup temporary clustered faces and output faces if requested
        if not fe_config.get("keep_intermediate_files", False):
            tools.remove_folder(fe_config.get("clusters_output_dir"))
            tools.remove_folder(fe_config.get("faces_output_dir"))
        tools.clear_memory()
        
        # 4. Photo enhancement
        print("✨ Enhancing faces...")
        model_type = fe_config.get("enhancement", {}).get("model", "gfpgan")
        model_path = fe_config.get("EDSR_model_path") if model_type == 'edsr' else fe_config.get("GFPGAN_model_path")
        upscale = fe_config.get("enhancement", {}).get("upscale", 4)

        enhancer.process_folders(
            best_faces_dir,
            model_type,
            model_path,
            upscale
        )

        print(f"✅ Processing complete! The result can be viewed in the directory {best_faces_dir}")
        
    except Exception as e:
        raise (f"❌ Error during processing: {e}")