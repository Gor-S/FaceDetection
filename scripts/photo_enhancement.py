import os
import cv2
import numpy as np
from gfpgan import GFPGANer

class FaceEnhancer:
    def __init__(self, model_type, model_path=None, upscale=2):
        """
        Initializes the FaceEnhancer class with the specified model type.
        
        Parameters:
        model_type (str): The type of model to use ('gfpgan', 'edsr', 'espcn').
        model_path (str, optional): Path to the model file. If None, default paths are used.
        upscale (int): Upscaling factor for the image.
        """
        self.model_type = model_type.lower()
        self.upscale = upscale

        if self.model_type == 'gfpgan':
            if model_path is None:
                model_path = './gfpgan/weights/gfpgan.pth'
            self.enhancer = GFPGANer(
                model_path=model_path,
                upscale=upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
        elif self.model_type == 'edsr':
            # Initialize OpenCV's super-resolution model
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            if model_path is None:
                model_path = f'./models/{self.model_type.upper()}_x2.pb'
            self.sr.readModel(model_path)
            self.sr.setModel(self.model_type, upscale)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def pre_process(self, image):
        """
        Applies Gaussian blur to the image and enhances sharpness.
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    def enhance(self, image):
        """
        Enhances the given image using the selected model.
        
        Parameters:
        image (numpy.ndarray): The input image.
        
        Returns:
        numpy.ndarray: The enhanced image.
        """
        if self.model_type == 'gfpgan':
            # Perform face restoration using GFPGAN
            _, _, restored_image = self.enhancer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True
            )
            return restored_image
        elif self.model_type == 'edsr':
            # Apply pre-processing and then super-resolution
            preprocessed = self.pre_process(image)
            return self.sr.upsample(preprocessed)
        return image

def process_folders(base_dir, model_type='gfpgan', model_path=None, upscale=2):
    """
    Processes all folders in the base directory and applies face enhancement.
    
    Parameters:
    base_dir (str): The base directory containing image folders.
    model_type (str, optional): The enhancement model to use.
    model_path (str, optional): Path to the model file.
    upscale (int, optional): Upscaling factor.
    """
    enhancer = FaceEnhancer(model_type=model_type, model_path=model_path, upscale=upscale)

    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("person_"):
            print(f"\n📂 Processing folder: {folder}")
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(folder_path, file)
                    output_path = os.path.join(folder_path, f"enhanced_{file}")

                    if os.path.exists(output_path):
                        print(f"⏩ Skipping: {output_path} (already exists)")
                        continue

                    image = cv2.imread(input_path)
                    if image is None:
                        print(f"⚠️ Skipping: {input_path} (failed to load)")
                        continue

                    # Enhance the image
                    enhanced_img = enhancer.enhance(image)
                    cv2.imwrite(output_path, enhanced_img)
                    print(f"✅ Saved: {output_path}")

