"""Face Finder Tools"""
import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video to determine FPS.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def seconds_to_timecode(seconds, fps=25):
    """
    Convert seconds to HH:MM:SS:FF timecode format
    
    Args:
        seconds (float): Time in seconds
        fps (float): Frames per second
    
    Returns:
        str: Formatted timecode
    """
    # Calculate hours, minutes, seconds, and frames
    total_frames = int(seconds * fps)
    frames = total_frames % int(fps)
    total_seconds = total_frames // int(fps)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    # Format as HH:MM:SS:FF
    return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

def extract_frame_number(filename):
    """
    Extracts the frame number from a filename of the format 'frame_<frame_number>_face_<n>.jpg'.
    """
    m = re.search(r"frame_(\d+)_face", filename)
    if m:
        return int(m.group(1))
    return None

def compute_embedding(image_path, resnet, transform, device):
    """
    Loads an image (face) from image_path, processes it, and computes its embedding.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to load image {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor)
    return embedding.cpu().numpy().flatten()

def process_extracted_faces(faces_dir, target_embedding, fps, threshold=0.6, min_gap_sec=1.0, device="cpu"):
    """
    Iterates over all saved face images in faces_dir, computes embeddings, and compares 
    them with the target_embedding. Returns a list of timestamps where matches were found,
    grouped into intervals.
    """
    # Define transformations and load the embedding model (similar to FaceClustering)
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    detected_times = []
    
    for file in os.listdir(faces_dir):
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        file_path = os.path.join(faces_dir, file)
        frame_number = extract_frame_number(file)
        if frame_number is None:
            continue
        try:
            embedding = compute_embedding(file_path, resnet, transform, device)
        except Exception as e:
            print(f"[ERROR] Error computing embedding for {file}: {e}")
            continue
        distance = np.linalg.norm(target_embedding - embedding)
        if distance < threshold:
            time_sec = frame_number / fps
            detected_times.append(time_sec)
    
    # Sort detected timestamps and group them into intervals
    detected_times.sort()
    intervals = []
    if detected_times:
        start = detected_times[0]
        prev = detected_times[0]
        for t in detected_times[1:]:
            if t - prev > min_gap_sec:
                intervals.append((start, prev))
                start = t
            prev = t
        intervals.append((start, prev))
    return intervals

def compute_target_embedding(target_face_path, device="cpu"):
    """
    Computes the embedding for the target face image.
    """
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])    # Resize enhanced image to match original if dimensions differ
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    img = cv2.imread(target_face_path)
    if img is None:
        raise ValueError("Unable to load target face image.")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor)
    return embedding.cpu().numpy().flatten()