import os
import cv2
from gfpgan import GFPGANer

class FaceRestorer:
    def __init__(self, model_path='./gfpgan/weights/gfpgan.pth', upscale=2):
        
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )

    def restore(self, input_path, output_path):
        image = cv2.imread(input_path)
        if image is None:
            print(f"⚠️ Skipping: {input_path} (not found or unreadable)")
            return

        _, _, restored_image = self.restorer.enhance(
            image,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        cv2.imwrite(output_path, restored_image)
        print(f"✅ Saved: {output_path}")

def process_folders(base_dir, model_path='./gfpgan/weights/gfpgan.pth'):
    restorer = FaceRestorer(model_path=model_path)

    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("person_"):
            print(f"\n📂 Processing folder: {folder}")

            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    input_path = os.path.join(folder_path, file)
                    output_path = os.path.join(folder_path, f"restored_{file}")
                    
                    # Skip if restored version already exists
                    if os.path.exists(output_path):
                        print(f"⏩ Skipping: {output_path} (already exists)")
                        continue

                    restorer.restore(input_path, output_path)
