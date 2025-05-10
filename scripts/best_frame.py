import os
import shutil
import numpy as np
import cv2
import mediapipe as mp
from . import logger
import sys
from .progress_bar import ProgressBar

# temporary solution to disable logs from absl, tensorflow at c++ level
null_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(null_fd, sys.stderr.fileno())

get_logger = logger.get_logger

class FaceEvaluator:
    def __init__(self, min_detection_confidence=0.3, eye_ar_base=0.15):
        self.min_detection_confidence = min_detection_confidence
        self.eye_ar_base = eye_ar_base
        self.face_detection_1 = mp.solutions.face_detection.FaceDetection(model_selection=1, 
                                                                          min_detection_confidence=min_detection_confidence)
        self.face_detection_0 = mp.solutions.face_detection.FaceDetection(model_selection=0, 
                                                                          min_detection_confidence=min_detection_confidence)
        # Initialize the face mesh model to detect facial landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        )
        self.logger = get_logger("face_evaluator")

    def evaluate_batch(self, image_paths):
        results = []
        with ProgressBar(total=len(image_paths), desc="Processing images", unit="img") as pbar:
            for image_path in image_paths:
                score = self.evaluate(image_path)
                results.append((image_path, score))
                pbar.update()
                sys.stdout.flush()
        return results

    def upscale_if_needed(self, image, min_size=150):
        h, w, _ = image.shape
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return image

    def evaluate_frontalness(self, image_rgb):
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return 0
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose_tip = landmarks[1]

        left_distance = abs(nose_tip.x - left_eye.x)
        right_distance = abs(right_eye.x - nose_tip.x)
        if max(left_distance, right_distance) < 1e-6:
            return 0
        ratio = min(left_distance, right_distance) / max(left_distance, right_distance)
        return ratio

    def evaluate_eyes_open(self, image_rgb):
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return 0
        landmarks = results.multi_face_landmarks[0].landmark

        # Left eye landmarks
        left_corner = landmarks[33]
        right_corner = landmarks[133]
        top_left = landmarks[159]
        bottom_left = landmarks[145]

        # Right eye landmarks
        right_corner_r = landmarks[263]
        left_corner_r = landmarks[362]
        top_right = landmarks[386]
        bottom_right = landmarks[374]

        # Calculate horizontal and vertical distances for left eye
        left_horizontal = np.linalg.norm([left_corner.x - right_corner.x, left_corner.y - right_corner.y])
        left_vertical = np.linalg.norm([top_left.x - bottom_left.x, top_left.y - bottom_left.y])
        left_ear = left_vertical / left_horizontal if left_horizontal > 0 else 0

        # Calculate horizontal and vertical distances for right eye
        right_horizontal = np.linalg.norm([right_corner_r.x - left_corner_r.x, right_corner_r.y - left_corner_r.y])
        right_vertical = np.linalg.norm([top_right.x - bottom_right.x, top_right.y - bottom_right.y])
        right_ear = right_vertical / right_horizontal if right_horizontal > 0 else 0

        # Average eye aspect ratio and normalize it
        ear_avg = (left_ear + right_ear) / 2
        normalized_ear = np.clip((ear_avg - self.eye_ar_base) / 0.2, 0, 1)
        return normalized_ear

    def evaluate(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            self.logger.debug(f"Could not load: {image_path}")
            return 0

        image = self.upscale_if_needed(image, min_size=150)
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.face_detection_1.process(image_rgb)
        if not results.detections:
            results0 = self.face_detection_0.process(image_rgb)
            if not results0.detections:
                self.logger.debug(f"Face not found: {image_path}")
                return 0
            else:
                detection = results0.detections[0]
        else:
            detection = results.detections[0]

        # Compute bounding box and face coverage
        bbox = detection.location_data.relative_bounding_box
        face_x, face_y = bbox.xmin * w, bbox.ymin * h
        face_w, face_h = bbox.width * w, bbox.height * h
        face_area = face_w * face_h
        coverage = face_area / (w * h)

        # Compute distance of face center from image center (normalized)
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2
        center_distance = np.sqrt((face_center_x - w / 2) ** 2 + (face_center_y - h / 2) ** 2)
        center_distance_norm = center_distance / np.sqrt((w / 2)**2 + (h / 2)**2)

        # Calculate sharpness using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_norm = np.clip(sharpness_val / 1000.0, 0, 1)

        # Calculate brightness score (how close average brightness is to 128)
        brightness = np.mean(gray)
        brightness_score = 1 - abs(brightness - 128) / 128
        brightness_score = np.clip(brightness_score, 0, 1)

        # Evaluate face frontalness and eye openness using face mesh
        frontalness = self.evaluate_frontalness(image_rgb)
        eyes_open_score = self.evaluate_eyes_open(image_rgb)
        resolution_score = np.clip((w * h) / 500000.0, 0, 1)

        # Weights for each metric
        w_sharp = 0.18
        w_bright = 0.16
        w_cov = 0.15
        w_center = 0.13
        w_frontal = 0.13
        w_eye = 0.13
        w_res = 0.12

        # Compute the final score as a weighted sum of all metrics
        final_score = (w_sharp * sharpness_norm +
                       w_bright * brightness_score +
                       w_cov * coverage +
                       w_center * (1 - center_distance_norm) +
                       w_frontal * frontalness +
                       w_eye * eyes_open_score +
                       w_res * resolution_score)

        self.logger.debug(f"{image_path}")
        self.logger.debug(f"  Sharpness={sharpness_val:.1f} => {sharpness_norm:.2f}")
        self.logger.debug(f"  Brightness={brightness:.1f} => {brightness_score:.2f}")
        self.logger.debug(f"  Coverage={coverage:.3f}")
        self.logger.debug(f"  CenterDist={center_distance_norm:.3f} => {1 - center_distance_norm:.3f}")
        self.logger.debug(f"  Frontalness={frontalness:.3f}")
        self.logger.debug(f"  EyesOpen={eyes_open_score:.3f}")
        self.logger.debug(f"  Resolution Score={(w * h):.0f} px => {resolution_score:.3f}")
        self.logger.debug(f"  => final_score={final_score:.3f}")

        return final_score


class FolderProcessor:
    @staticmethod
    def select_best_image_in_folder(folder_path, evaluator):
        logger = get_logger("folder_processor")
        best_score = -1
        best_image = None
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                score = evaluator.evaluate(image_path)
                if score > best_score:
                    best_score = score
                    best_image = image_path
        return best_image, best_score

    @staticmethod
    def process_all_folders(parent_folder, output_folder):
        logger = get_logger("folder_processor")
        os.makedirs(output_folder, exist_ok=True)
        evaluator = FaceEvaluator()
        for folder in os.listdir(parent_folder):
            folder_path = os.path.join(parent_folder, folder)
            if os.path.isdir(folder_path):
                best_image, score = FolderProcessor.select_best_image_in_folder(folder_path, evaluator)
                if best_image:
                    output_subfolder = os.path.join(output_folder, folder)
                    os.makedirs(output_subfolder, exist_ok=True)
                    output_path = os.path.join(output_subfolder, f"{folder}_best.jpg")
                    shutil.copy(best_image, output_path)
                    logger.info(f"Best photo in '{folder}': {os.path.basename(best_image)}, score={score:.2f} -> {output_path}")
                else:
                    logger.info(f"No suitable images found in '{folder}'")
