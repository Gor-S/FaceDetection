import os
import cv2
import gc
import torch
import multiprocessing
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

# Set device to CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

class FrameDataset(Dataset):
    def __init__(self, video_path, start_frame, end_frame, frame_skip):
        # Initialize dataset with video path, frame range, and frame skip interval
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_skip = frame_skip
        self.frames = self._extract_frames()

    def _extract_frames(self):
        # Extract frames from the video between start_frame and end_frame with frame skipping
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        frames = []
        frame_id = self.start_frame
        while cap.isOpened() and frame_id < self.end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % self.frame_skip == 0:
                frames.append((frame_id, frame))
            frame_id += 1
        cap.release()
        return frames

    def __len__(self):
        # Return the number of frames in the dataset
        return len(self.frames)

    def __getitem__(self, idx):
        # Return the frame corresponding to the given index
        return self.frames[idx]

def process_batch(model, batch, output_dir):
    try:
        # Batch is expected to be a list of tuples: (frame_id, frame)
        frame_ids, frames = zip(*batch)
        print(f"[DEBUG] Processing batch with {len(frames)} frames")
        # Convert frames from BGR to RGB – YOLO model expects images in RGB format
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        # Perform prediction; the predict method accepts a list of images
        results = model.predict(source=frames_rgb, device=device)
        # If the result is not a list, wrap it in a list
        if not isinstance(results, (list, tuple)):
            results = [results]
        for frame_id, frame, result in zip(frame_ids, frames, results):
            h, w, _ = frame.shape
            try:
                # Process all detected objects in the frame
                for i, box in enumerate(result.boxes.xyxy):
                    # Extract bounding box coordinates and add padding
                    x1, y1, x2, y2 = map(int, box)
                    padding_x, padding_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
                    x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                    x2, y2 = min(w, x2 + padding_x), min(h, y2 + padding_y)
                    face = frame[y1:y2, x1:x2]
                    # Unique filename for each face
                    save_path = os.path.join(output_dir, f"frame_{frame_id}_face_{i}.jpg")
                    cv2.imwrite(save_path, face)
            except Exception as inner_e:
                print(f"[ERROR] Processing result for frame {frame_id}: {type(inner_e).__name__}: {inner_e}")
    except Exception as e:
        # If an error occurs, e.g., AttributeError: bn, print the message and skip the batch
        print(f"[ERROR] Batch processing: {type(e).__name__}: {e}")

def process_range(video_path, start_frame, end_frame, frame_skip, output_dir, model_path, batch_size=8):
    # Load the model without the device argument, then move it to the appropriate device
    model = YOLO(model_path).to(device)
    if device == "cuda":
        model.fuse()  # Fuse model layers for CUDA
        model.half()  # Use FP16 for faster inference on CUDA

    dataset = FrameDataset(video_path, start_frame, end_frame, frame_skip)
    # Use a custom collate_fn to keep the batch as a list of tuples (avoiding automatic conversion to tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=lambda x: x)

    print(f"[{multiprocessing.current_process().name}] Processing frames {start_frame}–{end_frame}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch in dataloader:
            executor.submit(process_batch, model, batch, output_dir)
            gc.collect()  # Memory cleanup
            if device == "cuda":
                torch.cuda.empty_cache()

class VideoProcessor:
    def __init__(self, video_path, model_path, frame_skip, output_dir, max_gpu_workers=2, device=device):
        self.video_path = video_path
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.output_dir = output_dir
        self.device = device
        self.num_workers = self._choose_worker_count(max_gpu_workers)
        os.makedirs(self.output_dir, exist_ok=True)

    def _choose_worker_count(self, max_gpu_workers):
        # Get the number of CPU cores
        cpu_cores = multiprocessing.cpu_count()
        
        # Get the number of available GPUs
        gpu_cores = torch.cuda.device_count() if self.device == "cuda" else 0
        
        # Limit CPU workers to 70% of available cores
        cpu_limit = max(1, int(cpu_cores * 0.7))
        
        # Determine the number of workers based on GPU availability
        max_workers = gpu_cores * max_gpu_workers if gpu_cores > 0 else cpu_limit
        
        # Ensure that the number of workers does not exceed the CPU limit
        return min(cpu_limit, max_workers)


    def _split_video_ranges(self):
        # Split the video into ranges for parallel processing
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        part_len = total_frames // self.num_workers
        return [
            (i * part_len, (i + 1) * part_len if i < self.num_workers - 1 else total_frames)
            for i in range(self.num_workers)
        ]

    def process(self):
        # Start video processing
        print(f"[INFO] Device: {self.device.upper()}, Workers: {self.num_workers}")
        frame_ranges = self._split_video_ranges()
        with multiprocessing.get_context("spawn").Pool(processes=self.num_workers) as pool:
            args = [
                (self.video_path, start, end, self.frame_skip, self.output_dir, self.model_path)
                for start, end in frame_ranges
            ]
            pool.starmap(process_range, args)
        print("[INFO] Processing complete.")
