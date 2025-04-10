import os
import sys
import cv2
import gc
import torch
import multiprocessing
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics import settings
from . import tools
from . import logger
from . import progress_bar

logger.logging.getLogger("ultralytics").setLevel(logger.logging.ERROR)
device = "cuda"
config = tools.load_config("config/config.yaml")
get_logger = logger.get_logger

class FrameDataset(Dataset):
    def __init__(self, video_path, start_frame, end_frame, frame_skip):
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_skip = frame_skip
        self.logger = get_logger("frame_dataset")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"📂 File not found: {video_path}")

        self.cap = cv2.VideoCapture(self.video_path) 
        if not self.cap.isOpened():
            raise IOError(f"📹 Failed to open video: {video_path}")

        print("✅ [INFO] Video path is valid and accessible.")
        self.frames = self._extract_frames()  
    def _extract_frames(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        frames = []
        frame_id = self.start_frame
        
        while self.cap.isOpened() and frame_id < self.end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break
            if (frame_id - self.start_frame) % self.frame_skip == 0:
                frames.append((frame_id, frame))
            frame_id += 1
        self.cap.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

def process_batch(model, batch, output_dir, processed_faces_counter=None, frames_counter=None):
    logger = get_logger("process_batch")
    try:
        frame_ids, frames = zip(*batch)
        logger.debug(f"Processing batch with {len(frames)} frames")
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        results = model.predict(source=frames_rgb, device=device, verbose=False)

        face_count = 0
        if not isinstance(results, (list, tuple)):
            results = [results]
        for frame_id, frame, result in zip(frame_ids, frames, results):
            h, w, _ = frame.shape
            try:
                for i, box in enumerate(result.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    padding_x, padding_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
                    x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                    x2, y2 = min(w, x2 + padding_x), min(h, y2 + padding_y)
                    face = frame[y1:y2, x1:x2]
                    save_path = os.path.join(output_dir, f"frame_{frame_id}_face_{i}.jpg")
                    cv2.imwrite(save_path, face)
                    face_count += 1
            except Exception as inner_e:
                logger.error(f"Processing result for frame {frame_id}: {type(inner_e).__name__}: {inner_e}")
        
        if processed_faces_counter is not None:
            processed_faces_counter.value += face_count
        if frames_counter is not None:
            frames_counter.value += len(frames)
            
    except Exception as e:
        logger.error(f"Batch processing: {type(e).__name__}: {e}")


def collate_fn(batch):
    return batch

def process_range(video_path, start_frame, end_frame, frame_skip, output_dir, model_path, 
                  batch_size=8, num_workers=4, progress_shared_dict=None):
    logger = get_logger("process_range")

    if not torch.cuda.is_available():
        logger.error("❌ CUDA device not available. Cannot proceed with GPU-only execution.")
        raise RuntimeError("CUDA is not available on this system.")

    device = "cuda" 

    model = YOLO(model_path)
    model.to(device)

    model.fuse()
    model.half()

    dataset = FrameDataset(video_path, start_frame, end_frame, frame_skip)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info(f"[{multiprocessing.current_process().name}] Processing frames {start_frame}–{end_frame}")

    manager = multiprocessing.Manager()
    processed_faces_counter = manager.Value('i', 0)
    frames_counter = manager.Value('i', 0)

    # 🔁 Dataloader loop
    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch in dataloader:
            executor.submit(process_batch, model, batch, output_dir, processed_faces_counter, frames_counter)

            if progress_shared_dict is not None:
                process_id = multiprocessing.current_process().name
                progress_shared_dict[process_id] = {
                    'frames_processed': frames_counter.value,
                    'faces_detected': processed_faces_counter.value
                }

            gc.collect()
            torch.cuda.empty_cache()

    return processed_faces_counter.value
class VideoProcessor:
    def __init__(self, video_path, model_path, frame_skip, output_dir, max_gpu_workers=2, device=device):
        self.video_path = video_path
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.output_dir = output_dir
        self.device = device
        self.num_workers = self._choose_worker_count(max_gpu_workers)
        self.logger = get_logger("video_processor")
        os.makedirs(self.output_dir, exist_ok=True)

    def _choose_worker_count(self, max_gpu_workers):
        cpu_cores = multiprocessing.cpu_count()
        gpu_cores = torch.cuda.device_count() if self.device == "cuda" else 0
        
        # Limit CPU workers to 70% of available cores
        cpu_limit = max(1, int(cpu_cores * config["common"]["cpu_limit"]))
        
        max_workers = gpu_cores * max_gpu_workers if gpu_cores > 0 else cpu_limit
        
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
        self.logger.info(f"Device: {self.device.upper()}, Workers: {self.num_workers}")
        frame_ranges = self._split_video_ranges()
    
        # Calculate total frame count for progress reporting
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_total = total_frames // self.frame_skip
        cap.release()
    
        manager = multiprocessing.Manager()
        progress_shared_dict = manager.dict()
    
        with progress_bar.ProgressBar(total=effective_total, 
            desc="Video Processing", 
            unit="frames", 
            color="blue") as pbar:

            with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
                futures = [
                    executor.submit(
                        process_range, 
                        self.video_path, start, end, 
                        self.frame_skip, self.output_dir, self.model_path, 
                        num_workers=config["common"]["max_workers"],
                        progress_shared_dict=progress_shared_dict
                    )
                    for start, end in frame_ranges
                ]
            
                last_reported_progress = 0
                total_faces = 0
            
                update_interval = 0.05 
            
                while not all(f.done() for f in futures):
                    current_total_frames = sum(data.get('frames_processed', 0) 
                        for data in progress_shared_dict.values())
                    current_total_faces = sum(data.get('faces_detected', 0)
                        for data in progress_shared_dict.values())
                    new_frames = current_total_frames - last_reported_progress
                
                    # Update progress bar if there's new progress
                    if new_frames > 0:
                        pbar.update(new_frames)
                        last_reported_progress = current_total_frames
                    
                        # If faces count changed, add to description
                        if current_total_faces != total_faces:
                            total_faces = current_total_faces
                            pbar.set_description(f"Video Processing (Faces: {total_faces})")
                
                    sys.stdout.flush()
            
                    import time
                    time.sleep(update_interval)
            
                # Wait for all futures to complete and collect results
                total_faces_detected = sum(future.result() for future in futures)
        
            # Ensure the progress bar reaches 100% at the end
            if last_reported_progress < effective_total:
                pbar.update(effective_total - last_reported_progress)

        self.logger.info(f"Processing complete. Detected {total_faces_detected} faces in {effective_total} frames.")
