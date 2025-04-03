import os
import cv2
import numpy as np
from gfpgan import GFPGANer
from . import logger
from . import progress_bar

logger = logger.get_logger(module_name="photo_enhancement")

class FaceEnhancer:
    def __init__(self, model_type, model_path=None, upscale=2):
        """
        Initializes the FaceEnhancer class with the specified model type.
        
        Parameters:
        model_type (str): The type of model to use ('gfpgan', 'edsr').
        model_path (str, optional): Path to the model file. If None, default paths are used.
        upscale (int): Upscaling factor for the image.
        """
        self.model_type = model_type.lower()
        self.upscale = upscale
        logger.info(f"Initializing FaceEnhancer with model_type={model_type}, upscale={upscale}")

        if self.model_type == 'gfpgan':
            if model_path is None:
                model_path = './gfpgan/weights/gfpgan.pth'
            logger.debug(f"Loading GFPGAN model from {model_path}")
            self.enhancer = GFPGANer(
                model_path=model_path,
                upscale=upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
        elif self.model_type == 'edsr':
            # Initialize OpenCV's super-resolution model with upscale x4
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            if model_path is None:
                model_path = f'./models/{self.model_type.upper()}_x4.pb'
            logger.debug(f"Loading EDSR model from {model_path}")
            self.sr.readModel(model_path)
            self.sr.setModel(self.model_type, upscale)
        else:
            error_msg = f"Unsupported model type: {self.model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def pre_process(self, image):
        """
        Applies Gaussian blur to the image and enhances sharpness.
        """
        logger.debug("Pre-processing image with Gaussian blur and sharpness enhancement")
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
        logger.debug(f"Enhancing image using {self.model_type} model")
        if self.model_type == 'gfpgan':
            _, _, restored_image = self.enhancer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True
            )
            return restored_image
        elif self.model_type == 'edsr':
            preprocessed = self.pre_process(image)
            return self.sr.upsample(preprocessed)
        return image

def process_folders(base_dir, model_type='gfpgan', model_path=None, upscale=4): 
    """
    Processes all folders in the base directory and applies face enhancement.
    
    Parameters:
    base_dir (str): The base directory containing image folders.
    model_type (str, optional): The enhancement model to use.
    model_path (str, optional): Path to the model file.
    upscale (int, optional): Upscaling factor.
    """
    logger.info(f"Starting face enhancement process in {base_dir} with model_type={model_type}, upscale={upscale}")
    enhancer = FaceEnhancer(model_type=model_type, model_path=model_path, upscale=upscale)

    # Count total files for progress reporting
    total_files = 0
    person_folders = []
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("person_"):
            person_folders.append(folder)
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    total_files += 1
    
    with progress_bar.ProgressBar(total=total_files, desc="Enhancing faces", unit="images", color="green") as pbar:
        for folder in person_folders:
            folder_path = os.path.join(base_dir, folder)
            logger.info(f"Processing folder: {folder}")
            
            # Create folder progress bar
            folder_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in folder_files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(folder_path, file)
                    output_path = os.path.join(folder_path, f"enhanced_{file}")

                    if os.path.exists(output_path):
                        logger.debug(f"Skipping: {output_path} (already exists)")
                        pbar.update(1)
                        continue

                    image = cv2.imread(input_path)
                    if image is None:
                        logger.warning(f"Skipping: {input_path} (failed to load)")
                        pbar.update(1)
                        continue

                    # Enhance the image
                    enhanced_img = enhancer.enhance(image)
                    cv2.imwrite(output_path, enhanced_img)
                    logger.info(f"Saved: {output_path}")
                    pbar.update(1)
    
    logger.info("Face enhancement process completed")