import os
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
        self.frames = self._extract_frames()
        self.logger = get_logger("frame_dataset")

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
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

def process_batch(model, batch, output_dir):
    logger = get_logger("process_batch")
    try:
        frame_ids, frames = zip(*batch)
        logger.debug(f"Processing batch with {len(frames)} frames")
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        results = model.predict(source=frames_rgb, device=device, verbose=False)

        if not isinstance(results, (list, tuple)):
            results = [results]
        for frame_id, frame, result in zip(frame_ids, frames, results):
            h, w, _ = frame.shape
            try:
                for i, box in enumerate(result.boxes.xyxy):
                    # Extract bounding box coordinates and add padding
                    x1, y1, x2, y2 = map(int, box)
                    padding_x, padding_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
                    x1, y1 = max(0, x1 - padding_x), max(0, y1 - padding_y)
                    x2, y2 = min(w, x2 + padding_x), min(h, y2 + padding_y)
                    face = frame[y1:y2, x1:x2]
                    save_path = os.path.join(output_dir, f"frame_{frame_id}_face_{i}.jpg")
                    cv2.imwrite(save_path, face)
            except Exception as inner_e:
                logger.error(f"Processing result for frame {frame_id}: {type(inner_e).__name__}: {inner_e}")
    except Exception as e:
        logger.error(f"Batch processing: {type(e).__name__}: {e}")

def collate_fn(batch):
    return batch

def process_range(video_path, start_frame, end_frame, frame_skip, output_dir, model_path, batch_size=8, num_workers=4):
    logger = get_logger("process_range")
    model = YOLO(model_path).to(device)
    if device == "cuda":
        model.fuse()
        model.half()

    dataset = FrameDataset(video_path, start_frame, end_frame, frame_skip)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    logger.info(f"[{multiprocessing.current_process().name}] Processing frames {start_frame}–{end_frame}")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for batch in dataloader:
            executor.submit(process_batch, model, batch, output_dir)
            gc.collect()
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
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [
                executor.submit(process_range, self.video_path, start, end, self.frame_skip, self.output_dir, self.model_path, num_workers=config["common"]["max_workers"])
                for start, end in frame_ranges
            ]
            for future in futures:
                future.result() 

        self.logger.info("Processing complete.")