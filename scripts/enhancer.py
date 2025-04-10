import os
import cv2
import numpy as np
from gfpgan import GFPGANer
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
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


def calculate_improvement(original_img, enhanced_img):
    """
    Calculates the percentage of improvement between original and enhanced image.
    
    Parameters:
    original_img (numpy.ndarray): The original image.
    enhanced_img (numpy.ndarray): The enhanced image.
    
    Returns:
    dict: Dictionary containing different metrics and their improvement percentages.
    """
    # Resize enhanced image to match original if dimensions differ
    if original_img.shape != enhanced_img.shape:
        enhanced_img = cv2.resize(enhanced_img, (original_img.shape[1], original_img.shape[0]))
    
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    
    ssim_score = ssim(original_gray, enhanced_gray)
    
    psnr_score = psnr(original_img, enhanced_img)
    
    original_sharpness = cv2.Laplacian(original_gray, cv2.CV_64F).var()
    enhanced_sharpness = cv2.Laplacian(enhanced_gray, cv2.CV_64F).var()
    sharpness_improvement = (enhanced_sharpness - original_sharpness) / original_sharpness * 100
    
    normalized_psnr = min(100, max(0, (psnr_score - 20) * 100 / 30))

    overall_improvement = (
        0.4 * (ssim_score * 100) +  # 40% weight to SSIM
        0.4 * normalized_psnr +      # 40% weight to PSNR
        0.2 * max(-100, min(100, sharpness_improvement))  # 20% weight to sharpness, capped at ±100%
    )
    
    return {
        "ssim": ssim_score * 100,  # Convert to percentage
        "psnr": psnr_score,
        "sharpness_improvement": sharpness_improvement,
        "overall_improvement": overall_improvement
    }


def process_folders(base_dir, model_type='gfpgan', model_path=None, upscale=4, save_metrics=True): 
    """
    Processes all folders in the base directory and applies face enhancement.
    
    Parameters:
    base_dir (str): The base directory containing image folders.
    model_type (str, optional): The enhancement model to use.
    model_path (str, optional): Path to the model file.
    upscale (int, optional): Upscaling factor.
    save_metrics (bool, optional): Whether to save metrics to a CSV file.
    """
    logger.info(f"Starting face enhancement process in {base_dir} with model_type={model_type}, upscale={upscale}")
    enhancer = FaceEnhancer(model_type=model_type, model_path=model_path, upscale=upscale)
    
    metrics_file = None
    if save_metrics:
        metrics_path = os.path.join(base_dir, "enhancement_metrics.csv")
        metrics_file = open(metrics_path, 'w')
        metrics_file.write("folder,file,ssim,psnr,sharpness_improvement,overall_improvement\n")

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
            
            folder_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for file in folder_files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_path = os.path.join(folder_path, file)
                    output_path = os.path.join(folder_path, f"enhanced_{file}")
                    metrics_path = os.path.join(folder_path, f"metrics_{file.split('.')[0]}.txt")

                    if os.path.exists(output_path):
                        logger.debug(f"Skipping: {output_path} (already exists)")
                        pbar.update(1)
                        continue

                    original_img = cv2.imread(input_path)
                    if original_img is None:
                        logger.warning(f"Skipping: {input_path} (failed to load)")
                        pbar.update(1)
                        continue

                    enhanced_img = enhancer.enhance(original_img)
                    
                    metrics = calculate_improvement(original_img, enhanced_img)
                    
                    cv2.imwrite(output_path, enhanced_img)
                    
                    # Save metrics to a text file for each image
                    with open(metrics_path, 'w') as f:
                        f.write(f"Image Quality Improvement Metrics for {file}:\n")
                        f.write(f"SSIM: {metrics['ssim']:.2f}%\n")
                        f.write(f"PSNR: {metrics['psnr']:.2f} dB\n")
                        f.write(f"Sharpness Improvement: {metrics['sharpness_improvement']:.2f}%\n")
                        f.write(f"Overall Improvement: {metrics['overall_improvement']:.2f}%\n")
                    
                    if metrics_file:
                        metrics_file.write(f"{folder},{file},{metrics['ssim']:.2f},{metrics['psnr']:.2f},"
                                          f"{metrics['sharpness_improvement']:.2f},{metrics['overall_improvement']:.2f}\n")
                    
                    logger.info(f"Saved: {output_path} (Improvement: {metrics['overall_improvement']:.2f}%)")
                    pbar.update(1)

    if metrics_file:
        metrics_file.close()
        logger.info(f"Saved metrics to {os.path.join(base_dir, 'enhancement_metrics.csv')}")
    
    logger.info("Face enhancement process completed")


def generate_summary_report(base_dir):
    """
    Generates a summary report of all enhancement metrics.
    
    Parameters:
    base_dir (str): The base directory containing image folders.
    
    Returns:
    str: Path to the generated report file.
    """
    metrics_path = os.path.join(base_dir, "enhancement_metrics.csv")
    summary_path = os.path.join(base_dir, "enhancement_summary.txt")
    
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return None
    
    import csv
    metrics_data = []
    with open(metrics_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics_data.append({
                'folder': row['folder'],
                'file': row['file'],
                'ssim': float(row['ssim']),
                'psnr': float(row['psnr']),
                'sharpness_improvement': float(row['sharpness_improvement']),
                'overall_improvement': float(row['overall_improvement'])
            })
    
    with open(summary_path, 'w') as f:
        f.write("# Image Enhancement Summary Report\n\n")
        
        avg_improvement = sum(item['overall_improvement'] for item in metrics_data) / len(metrics_data)
        max_improvement = max(metrics_data, key=lambda x: x['overall_improvement'])
        min_improvement = min(metrics_data, key=lambda x: x['overall_improvement'])
        
        f.write(f"Total images processed: {len(metrics_data)}\n")
        f.write(f"Average overall improvement: {avg_improvement:.2f}%\n")
        f.write(f"Highest improvement: {max_improvement['overall_improvement']:.2f}% ({max_improvement['folder']}/{max_improvement['file']})\n")
        f.write(f"Lowest improvement: {min_improvement['overall_improvement']:.2f}% ({min_improvement['folder']}/{min_improvement['file']})\n\n")
        
        folders = {}
        for item in metrics_data:
            folder = item['folder']
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(item)
        
        f.write("## Per-folder Statistics\n\n")
        for folder, items in sorted(folders.items()):
            avg_folder_improvement = sum(item['overall_improvement'] for item in items) / len(items)
            f.write(f"### {folder}\n")
            f.write(f"Images processed: {len(items)}\n")
            f.write(f"Average improvement: {avg_folder_improvement:.2f}%\n\n")
            
            f.write("| Image | Overall Improvement |\n")
            f.write("|-------|---------------------|\n")
            for item in sorted(items, key=lambda x: x['overall_improvement'], reverse=True):
                f.write(f"| {item['file']} | {item['overall_improvement']:.2f}% |\n")
            f.write("\n")
    
    logger.info(f"Generated summary report: {summary_path}")
    return summary_path