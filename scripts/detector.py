import os
import sys
import cv2
import gc
import torch
import psutil
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
device = "cuda" if torch.cuda.is_available() else "cpu"
config = tools.load_config("config/config.yaml")
get_logger = logger.get_logger

class FrameDataset(Dataset):
    def __init__(self, video_path, start_frame, end_frame, frame_skip):
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_skip = frame_skip
        self.cap = None
        self.frame_indices = self._calculate_frame_indices()
        self.logger = get_logger("frame_dataset")
        self._open_video()

    def _open_video(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open video file: {self.video_path}")
        except Exception as e:
            self.logger.error(f"Error opening video: {e}")
            self.cap = None

    def _calculate_frame_indices(self):
        return [frame_id for frame_id in range(self.start_frame, self.end_frame) 
                if (frame_id - self.start_frame) % self.frame_skip == 0]

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        frame_id = self.frame_indices[idx]
        try:
            if self.cap is None or not self.cap.isOpened():
                self._open_video()
                
            if self.cap is None or not self.cap.isOpened():
                self.logger.error(f"Cannot read from video file for frame {frame_id}")
                return frame_id, np.zeros((32, 32, 3), dtype=np.uint8)
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning(f"Failed to read frame {frame_id}, returning blank frame")
                return frame_id, np.zeros((32, 32, 3), dtype=np.uint8)
            
            # Return a copy to prevent reference issues
            return frame_id, frame.copy()
        except Exception as e:
            self.logger.error(f"Error reading frame {frame_id}: {type(e).__name__}: {e}")
            return frame_id, np.zeros((32, 32, 3), dtype=np.uint8)
    
    def __del__(self):
        try:
            if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            self.logger.error(f"Error closing video in destructor: {e}")

def get_memory_usage():
    """Get current memory usage in MB for logging"""
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)
    
    gpu_usage = 0
    if device == "cuda" and torch.cuda.is_available():
        gpu_usage = torch.cuda.memory_allocated() / (1024 * 1024)
    
    return ram_usage, gpu_usage

def process_batch(model, batch, output_dir, processed_faces_counter=None, frames_counter=None):
    logger = get_logger("process_batch")
    try:
        frame_ids, frames = zip(*batch)
        logger.debug(f"Processing batch with {len(frames)} frames")
        
        ram_usage, gpu_usage = get_memory_usage()
        logger.debug(f"Memory before detection: RAM: {ram_usage:.2f} MB, GPU: {gpu_usage:.2f} MB")
        
        total_faces = 0
        for frame_id, frame in zip(frame_ids, frames):
            try:
                # Skip tiny frames (likely failed reads)
                if frame.shape[0] < 32 or frame.shape[1] < 32:
                    logger.warning(f"Skipping too small frame: {frame.shape}")

                    if frames_counter is not None:
                        frames_counter.value += 1
                        
                    continue
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = model.predict(source=frame_rgb, device=device, verbose=False)[0]
                
                h, w, _ = frame.shape
                valid_boxes = result.boxes[result.boxes.conf > 0.5].xyxy
                for i, box in enumerate(valid_boxes):
                    x1, y1, x2, y2 = map(int, box)
                    padding_x, padding_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
                    x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                    x2, y2 = min(w, x2 + padding_x), min(h, y2 + padding_y)
                    
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue
                        
                    face = frame[y1:y2, x1:x2]
                    save_path = os.path.join(output_dir, f"frame_{frame_id}_face_{i}.jpg")
                    
                    cv2.imwrite(save_path, face, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    total_faces += 1
                
                del result, frame_rgb
                
                if frames_counter is not None:
                    frames_counter.value += 1
                    
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as inner_e:
                logger.error(f"Processing frame {frame_id}: {type(inner_e).__name__}: {inner_e}")
                if frames_counter is not None:
                    frames_counter.value += 1
            
            # Clear frame from memory
            del frame
            
        if processed_faces_counter is not None:
            processed_faces_counter.value += total_faces
            
        ram_usage, gpu_usage = get_memory_usage()
        logger.debug(f"Memory after detection: RAM: {ram_usage:.2f} MB, GPU: {gpu_usage:.2f} MB")
            
    except Exception as e:
        logger.error(f"Batch processing: {type(e).__name__}: {e}")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def collate_fn(batch):
    return batch

def process_range(video_path, start_frame, end_frame, frame_skip, output_dir, model_path, 
                  batch_size=4, num_workers=0, progress_shared_dict=None):
    logger = get_logger("process_range")
    
    try:
        model = YOLO(model_path).to(device)
        if device == "cuda":
            # Only fuse if on CUDA
            model.fuse()
            vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            logger.info(f"Available VRAM: {vram_mb:.2f} MB")
            
            if vram_mb > 4 * 1024:  # > 4GB
                logger.info("Using mixed precision (FP16)")
                model.half()
            else:
                logger.info("Using full precision (FP32) due to limited VRAM")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file for dimension check: {video_path}")
            cap.release()
            return 0
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        pixel_count = frame_width * frame_height
        original_batch_size = batch_size
        
        if pixel_count > 1280 * 720:  # > 720p
            batch_size = max(1, batch_size // 2)
        if pixel_count > 1920 * 1080:  # > 1080p
            batch_size = max(1, batch_size // 4)
            
        if original_batch_size != batch_size:
            logger.info(f"Resolution-based batch size adjustment: {original_batch_size} → {batch_size} ({frame_width}x{frame_height})")
        
        dataset = FrameDataset(video_path, start_frame, end_frame, frame_skip)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=0, collate_fn=collate_fn, pin_memory=False)

        logger.info(f"[{multiprocessing.current_process().name}] Processing frames {start_frame}–{end_frame}, batch size: {batch_size}")

        manager = multiprocessing.Manager()
        processed_faces_counter = manager.Value('i', 0)
        frames_counter = manager.Value('i', 0)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            for batch in dataloader:
                ram_usage, gpu_usage = get_memory_usage()
                logger.debug(f"Memory before batch: RAM: {ram_usage:.2f} MB, GPU: {gpu_usage:.2f} MB")
                
                process_batch(model, batch, output_dir, processed_faces_counter, frames_counter)
                
                if progress_shared_dict is not None:
                    process_id = multiprocessing.current_process().name
                    progress_shared_dict[process_id] = {
                        'frames_processed': frames_counter.value,
                        'faces_detected': processed_faces_counter.value,
                        'ram_usage': ram_usage,
                        'gpu_usage': gpu_usage if device == "cuda" else 0
                    }

                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
        
        # Clean up
        del model
        del dataset
        del dataloader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return processed_faces_counter.value
        
    except Exception as e:
        logger.error(f"Error in process_range: {type(e).__name__}: {e}")
        return 0


class VideoProcessor:
    def __init__(self, video_path, model_path, frame_skip, output_dir, max_gpu_workers=1, device=device):
        self.video_path = video_path
        self.model_path = model_path
        self.frame_skip = frame_skip
        self.output_dir = output_dir
        self.device = device
        self.logger = get_logger("video_processor")
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating output directory: {e}")
            
        try:
            self.video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            self.logger.info(f"Video file size: {self.video_size_mb:.2f} MB")
        except Exception as e:
            self.logger.error(f"Error getting video file size: {e}")
            self.video_size_mb = 0
            
        self.num_workers = self._choose_worker_count(max_gpu_workers)

    def _choose_worker_count(self, max_gpu_workers):
        """Choose optimal worker count based on system resources and file size"""
        cpu_cores = multiprocessing.cpu_count()
        gpu_cores = torch.cuda.device_count() if self.device == "cuda" else 0
        
        # More conservative worker allocation
        cpu_limit_factor = config["common"]["cpu_limit"]
        file_size_mb = self.video_size_mb
        
        # Check if video is large
        is_large_file = file_size_mb > 50
        is_very_large_file = file_size_mb > 100
        
        # Adjust worker count based on file size
        if is_large_file:
            cpu_limit_factor *= 0.5  # More aggressive reduction (50%)
        if is_very_large_file:
            cpu_limit_factor *= 0.5  
            
        # Calculate CPU and GPU worker limits
        cpu_limit = max(1, int(cpu_cores * cpu_limit_factor))
        
        # For large files, restrict to 1 GPU worker
        gpu_worker_limit = 1 if is_large_file else max_gpu_workers
        
        # Calculate total workers
        if gpu_cores > 0:
            max_workers = min(gpu_cores * gpu_worker_limit, cpu_limit)
        else:
            max_workers = cpu_limit
        
        if is_very_large_file:
            max_workers = 1
            
        # Log worker selection decision
        self.logger.info(f"Worker selection: CPU cores={cpu_cores}, GPU cores={gpu_cores}, "
                         f"File size={file_size_mb:.2f}MB → {max_workers} workers")
            
        return max(1, max_workers)

    def _split_video_ranges(self):
        """Split video into processing ranges based on file size and worker count"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video file for range splitting: {self.video_path}")
                return [(0, 1000)]  # Return a fallback range
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            self.logger.info(f"Video has {total_frames} frames at {fps} FPS")

            if self.num_workers == 1 or self.video_size_mb <= 50:
                self.logger.info("Using single range for entire video")
                return [(0, total_frames)]
            
            # For multiple workers on larger files, use time-based chunks
            chunk_duration = 20  # seconds (reduced from 30)
            frames_per_chunk = int(fps * chunk_duration)
            
            # Create initial ranges
            ranges = []
            start = 0
            while start < total_frames:
                end = min(start + frames_per_chunk, total_frames)
                ranges.append((start, end))
                start = end
            
            # Consolidate ranges if there are too many compared to workers
            if len(ranges) > self.num_workers:
                self.logger.info(f"Consolidating {len(ranges)} ranges to {self.num_workers} workers")
                
                while len(ranges) > self.num_workers:
                    ranges.sort(key=lambda x: x[1] - x[0])  
                    shortest = ranges.pop(0) 
                    next_shortest = ranges.pop(0)
                    ranges.append((shortest[0], next_shortest[1]))  
            
            self.logger.info(f"Split video into {len(ranges)} chunks")
            return ranges
            
        except Exception as e:
            self.logger.error(f"Error splitting video ranges: {e}")
  
            return [(0, 1000)]

    def process(self):
        """Process video and extract faces"""
        self.logger.info(f"Starting video processing: Device: {self.device.upper()}, "
                        f"Workers: {self.num_workers}, File size: {self.video_size_mb:.2f} MB")
        
        try:
            frame_ranges = self._split_video_ranges()
        
            # Get video info for progress tracking
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video file: {self.video_path}")
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            effective_total = max(1, total_frames // self.frame_skip)
            cap.release()
        
            manager = multiprocessing.Manager()
            progress_shared_dict = manager.dict()
        
            # Initialize progress bar
            with progress_bar.ProgressBar(total=effective_total, 
                desc=f"Processing {os.path.basename(self.video_path)}", 
                unit="frames", 
                color="blue") as pbar:

                # Log initial memory usage
                ram_usage, gpu_usage = get_memory_usage()
                self.logger.info(f"Initial memory: RAM: {ram_usage:.2f} MB, GPU: {gpu_usage:.2f} MB")

                # Calculate batch size based on video size
                batch_size = 4 
                if self.video_size_mb > 50:
                    batch_size = 2
                if self.video_size_mb > 100:
                    batch_size = 1
                
                use_sequential = self.video_size_mb > 50 or self.num_workers == 1
                
                if use_sequential:
                    self.logger.info(f"Using sequential processing for {len(frame_ranges)} chunks")
                    # Process ranges one by one
                    total_faces_detected = 0
                    current_progress = 0
                    
                    for i, (start, end) in enumerate(frame_ranges):
                        self.logger.info(f"Processing chunk {i+1}/{len(frame_ranges)}: frames {start}-{end}")
                        
                        try:
                            faces = process_range(
                                self.video_path, start, end, 
                                self.frame_skip, self.output_dir, self.model_path, 
                                batch_size=batch_size,
                                progress_shared_dict=progress_shared_dict
                            )
                            total_faces_detected += faces
                            
                            current_total_frames = sum(data.get('frames_processed', 0) 
                                for data in progress_shared_dict.values())
                            frames_to_update = min(current_total_frames - current_progress, effective_total - current_progress)
                            pbar.update(frames_to_update)
                            current_progress = current_total_frames
                            
                            gc.collect()
                            if self.device == "cuda":
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            self.logger.error(f"Error processing chunk {i+1}: {e}")
                else:
                    self.logger.info(f"Using parallel processing with {self.num_workers} workers")
                    # Use process pool for parallel processing
                    with ProcessPoolExecutor(max_workers=self.num_workers, 
                                           mp_context=multiprocessing.get_context("spawn")) as executor:
                        futures = [
                            executor.submit(
                                process_range, 
                                self.video_path, start, end, 
                                self.frame_skip, self.output_dir, self.model_path, 
                                batch_size=batch_size,
                                progress_shared_dict=progress_shared_dict
                            )
                            for start, end in frame_ranges
                        ]
                    
                        last_reported_progress = 0
                        current_progress = 0
                        total_faces = 0
                    
                        update_interval = 0.5 
                    
                        # Monitor progress while processing
                        while not all(f.done() for f in futures):
                            try:
                                current_total_frames = sum(data.get('frames_processed', 0) 
                                    for data in progress_shared_dict.values())
                                current_total_faces = sum(data.get('faces_detected', 0)
                                    for data in progress_shared_dict.values())
                                
                                # Calculate current memory usage
                                current_ram = sum(data.get('ram_usage', 0) 
                                    for data in progress_shared_dict.values())
                                current_gpu = sum(data.get('gpu_usage', 0)
                                    for data in progress_shared_dict.values())
                                    
                                new_frames = current_total_frames - last_reported_progress
                            
                                if new_frames > 0:
                                    frames_to_update = min(new_frames, effective_total - current_progress)
                                    pbar.update(frames_to_update)
                                    current_progress += frames_to_update
                                    last_reported_progress = current_total_frames
                                
                                    if current_total_faces != total_faces:
                                        total_faces = current_total_faces
                                        pbar.set_description(
                                            f"Processing ({total_faces} faces | RAM: {current_ram:.0f}MB | GPU: {current_gpu:.0f}MB)"
                                        )
                            
                                sys.stdout.flush()
                            
                            except Exception as e:
                                self.logger.error(f"Error monitoring progress: {e}")
                        
                            import time
                            time.sleep(update_interval)

                        total_faces_detected = 0
                        for future in futures:
                            try:
                                total_faces_detected += future.result()
                            except Exception as e:
                                self.logger.error(f"Error getting future result: {e}")
                
                    if current_progress < effective_total:
                        pbar.update(effective_total - current_progress)

            ram_usage, gpu_usage = get_memory_usage()
            self.logger.info(f"Final memory: RAM: {ram_usage:.2f} MB, GPU: {gpu_usage:.2f} MB")
            self.logger.info(f"Processing complete. Detected {total_faces_detected} faces in {effective_total} frames.")
            
        except Exception as e:
            self.logger.error(f"Error in main processing loop: {type(e).__name__}: {e}")