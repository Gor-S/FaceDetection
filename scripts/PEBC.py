"""Photo enhancement before clustering"""
import cv2
import os
import numpy as np
from . import logger

logger = logger.get_logger(module_name="PEBC")

def enhance_image(image):
    # 1. Remove noise using Gaussian Blur
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # 2. Enhance contrast using CLAHE (based on the color image)
    lab = cv2.cvtColor(image_blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    image_denoised = cv2.medianBlur(image_clahe, 3)
    
    # 4. Apply sharpening using Unsharp Mask
    blurred = cv2.GaussianBlur(image_denoised, (9, 9), 10.0)
    improved_image = cv2.addWeighted(image_denoised, 1.5, blurred, -0.5, 0)
    
    return improved_image

def process_images_in_folder(folder_path):

    logger.info(f"Starting image processing in folder: {folder_path}")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Load the image
            image = cv2.imread(file_path)
            
            if image is not None:
                logger.debug(f"Enhancing image: {filename}")
                enhanced_image = enhance_image(image)

                cv2.imwrite(file_path, enhanced_image)
                logger.info(f"Image {filename} enhanced and saved.")
            else:
                logger.error(f"Failed to load {filename}.")
        else:
            logger.debug(f"{filename} is not an image.")
    
    logger.info(f"Completed processing all images in folder: {folder_path}")