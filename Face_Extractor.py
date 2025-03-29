from scripts import detect_faces, grouping_of_persons, choosing_the_best_frame, photo_enhancement, PEBC, tools

if __name__ == "__main__":
    
    try:
        # 1. Face detection
        print("🔍 Detecting faces...")
        config = tools.load_config("FE-config.yaml")
        processor = detect_faces.VideoProcessor(
            config["video_path"], 
            config["face_model_path"],
            config["frame_skip"],
            config["faces_output_dir"]
        )
        processor.process()
        tools.clear_memory()
    
        # 2. Face clustering
        PEBC.process_images_in_folder(config["faces_output_dir"])
        print("📂 Clustering faces...")
        clusterer = grouping_of_persons.FaceClustering(
            config["faces_output_dir"], 
            config["clusters_output_dir"]
        )
        clusterer.cluster_faces()
        tools.clear_memory()
        
        # 3. Selecting the best frames
        print("📸 Selecting best frames...")
        choosing_the_best_frame.FolderProcessor.process_all_folders(
            config["clusters_output_dir"], 
            config["best_faces_dir"]
        )

        # Cleanup temporary clustered faces and output faces
        tools.remove_folder(config["clusters_output_dir"])
        tools.remove_folder(config["faces_output_dir"])
        tools.clear_memory()
        
        # 4. Photo enhancement
        print("✨ Enhancing faces...")
        model_type = 'gfpgan'  # gfpgan or edsr
        model_path = config["EDSR_model_path"] if model_type == 'edsr' else config["GFPGAN_model_path"]

        photo_enhancement.process_folders(
            config["best_faces_dir"],
            model_type,
            model_path
        )

        print("✅ Processing complete!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    