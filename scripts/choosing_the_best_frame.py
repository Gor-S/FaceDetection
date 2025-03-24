import cv2
import mediapipe as mp
import os
import numpy as np
import shutil

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def clear_clustered_faces(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

class FaceEvaluator:
    @staticmethod
    def upscale_if_needed(image, min_size=150):
        """
        If the height or width of the face (or the entire image) is too small,
        upscale the image to increase the likelihood of correct recognition by Mediapipe.

        min_size: the minimum height or width below which upscaling is performed.
        """
        h, w, _ = image.shape
        if h < min_size or w < min_size:
            # Upscaling factor:
            # e.g., if h=75 and min_size=150, scale ~ 2
            scale = max(min_size / h, min_size / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return image

    @staticmethod
    def evaluate_frontalness(image_rgb, min_detection_confidence=0.3):
        """
        Evaluates the "frontalness" of a face: how centered the nose is between the eyes.
        Returns a value between 0 and 1, where 1 means a perfectly frontal face.
        """
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return 0
            landmarks = results.multi_face_landmarks[0].landmark

            # Indices: 1 - nose, 33 - left outer eye corner, 263 - right outer eye corner
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            nose_tip = landmarks[1]

            left_distance = abs(nose_tip.x - left_eye.x)
            right_distance = abs(right_eye.x - nose_tip.x)
            if max(left_distance, right_distance) < 1e-6:
                return 0
            ratio = min(left_distance, right_distance) / max(left_distance, right_distance)
            return ratio

    @staticmethod
    def evaluate_eyes_open(image_rgb, min_detection_confidence=0.3, eye_ar_base=0.15):
        """
        Evaluates how open the eyes are (using Eye Aspect Ratio, EAR).
        Returns a value between 0 and 1, where 1 means fully open eyes.

        The parameter eye_ar_base sets the baseline EAR threshold for normalization.
        The lower the eye_ar_base, the easier it is to "pass" the open-eyes check.
        """
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence
        ) as face_mesh:
            results = face_mesh.process(image_rgb)
            if not results.multi_face_landmarks:
                return 0
            landmarks = results.multi_face_landmarks[0].landmark

            # Left eye (outer and inner corners, top and bottom points)
            left_corner = landmarks[33]
            right_corner = landmarks[133]
            top_left = landmarks[159]
            bottom_left = landmarks[145]

            # Right eye (outer and inner corners, top and bottom points)
            right_corner_r = landmarks[263]
            left_corner_r = landmarks[362]
            top_right = landmarks[386]
            bottom_right = landmarks[374]

            # EAR for left eye
            left_horizontal = np.sqrt((left_corner.x - right_corner.x)**2 + (left_corner.y - right_corner.y)**2)
            left_vertical = np.sqrt((top_left.x - bottom_left.x)**2 + (top_left.y - bottom_left.y)**2)
            left_ear = left_vertical / left_horizontal if left_horizontal > 0 else 0

            # EAR for right eye
            right_horizontal = np.sqrt((right_corner_r.x - left_corner_r.x)**2 + (right_corner_r.y - left_corner_r.y)**2)
            right_vertical = np.sqrt((top_right.x - bottom_right.x)**2 + (top_right.y - bottom_right.y)**2)
            right_ear = right_vertical / right_horizontal if right_horizontal > 0 else 0

            ear_avg = (left_ear + right_ear) / 2

            # Normalization:
            # For example, if ear_avg = 0.15 and eye_ar_base = 0.15, then normalized_ear = (0.15 - 0.15)/0.2 = 0
            # If ear_avg = 0.25, normalized_ear = (0.25 - 0.15)/0.2 = 0.5
            # The higher the ear_avg, the closer to 1.
            normalized_ear = np.clip((ear_avg - eye_ar_base) / 0.2, 0, 1)
            return normalized_ear

    @classmethod
    def evaluate_face_quality(cls, image_path):
        """
        Calculates the final "rating" of an image based on:
          1. Sharpness.
          2. Brightness.
          3. Face coverage in the frame.
          4. Centering of the face.
          5. Frontalness.
          6. Eye openness.
          7. Resolution.

        Returns a number between 0 and 1 (approximately), where higher is better.
        If no face is found, returns 0.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"[DEBUG] Could not load: {image_path}")
            return 0

        # Upscale the image if it is too small:
        image = cls.upscale_if_needed(image, min_size=150)

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Face detection (lower the threshold to 0.3 to not miss weak faces)
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
            results = face_detection.process(image_rgb)
            if not results.detections:
                # If no face is found, try again with model 0 (sometimes better for large faces)
                with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face_detection0:
                    results0 = face_detection0.process(image_rgb)
                    if not results0.detections:
                        print(f"[DEBUG] Face not found: {image_path}")
                        return 0
                    else:
                        detection = results0.detections[0]
            else:
                detection = results.detections[0]

        # Get the face bounding box
        bbox = detection.location_data.relative_bounding_box
        face_x, face_y = bbox.xmin * w, bbox.ymin * h
        face_w, face_h = bbox.width * w, bbox.height * h
        face_area = face_w * face_h
        coverage = face_area / (w * h)  # ratio of face area to total image area

        # Calculate the center of the face
        face_center_x = face_x + face_w / 2
        face_center_y = face_y + face_h / 2
        center_distance = np.sqrt((face_center_x - w / 2) ** 2 + (face_center_y - h / 2) ** 2)
        center_distance_norm = center_distance / np.sqrt((w / 2)**2 + (h / 2)**2)  # 0 => centered, 1 => in the corner

        # Other metrics
        # Sharpness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize sharpness (e.g., 0..1000 => 0..1)
        sharpness_norm = np.clip(sharpness_val / 1000.0, 0, 1)

        # Brightness
        brightness = np.mean(gray)
        # Convert to a score in the range 0..1: the closer to 128, the better
        brightness_score = 1 - abs(brightness - 128) / 128
        brightness_score = np.clip(brightness_score, 0, 1)

        # Frontalness
        frontalness = cls.evaluate_frontalness(image_rgb, min_detection_confidence=0.3)

        # Eye openness
        eyes_open_score = cls.evaluate_eyes_open(image_rgb, min_detection_confidence=0.3, eye_ar_base=0.15)

        # Resolution score:
        # We normalize the resolution based on a threshold.
        # For example, if we set 500,000 pixels as the threshold, then:
        resolution_score = np.clip((w * h) / 500000.0, 0, 1)

        # Weight coefficients (adjustable)
        w_sharp = 0.18
        w_bright = 0.18
        w_cov = 0.13
        w_center = 0.13
        w_frontal = 0.13
        w_eye = 0.13
        w_res = 0.12

        # Final metric:
        # (1 - center_distance_norm), since 0 => centered, 1 => far => the lower, the better.
        final_score = (w_sharp * sharpness_norm +
                       w_bright * brightness_score +
                       w_cov * coverage +
                       w_center * (1 - center_distance_norm) +
                       w_frontal * frontalness +
                       w_eye * eyes_open_score +
                       w_res * resolution_score)

        # [DEBUG] Optionally print to see detailed reasons
        # (Disable if you do not want debug logs)
        print(f"[DEBUG] {image_path}")
        print(f"  Sharpness={sharpness_val:.1f} => {sharpness_norm:.2f}")
        print(f"  Brightness={brightness:.1f} => {brightness_score:.2f}")
        print(f"  Coverage={coverage:.3f}")
        print(f"  CenterDist={center_distance_norm:.3f} => {1 - center_distance_norm:.3f}")
        print(f"  Frontalness={frontalness:.3f}")
        print(f"  EyesOpen={eyes_open_score:.3f}")
        print(f"  Resolution Score={(w * h):.0f} px => {resolution_score:.3f}")
        print(f"  => final_score={final_score:.3f}\n")

        return final_score

class FolderProcessor:
    @staticmethod
    def select_best_image_in_folder(folder_path):
        """
        Iterates over all images in a folder and selects the one with the highest final score.
        """
        best_score = -1
        best_image = None
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                score = FaceEvaluator.evaluate_face_quality(image_path)
                if score > best_score:
                    best_score = score
                    best_image = image_path
        return best_image, best_score

    @staticmethod
    def process_all_folders(parent_folder, output_folder):
        """
        Iterates over all subfolders of parent_folder (person_0, person_1, ...),
        finds the best photo in each folder, and copies it to output_folder.
        """
        os.makedirs(output_folder, exist_ok=True)
        for folder in os.listdir(parent_folder):
            folder_path = os.path.join(parent_folder, folder)
            if os.path.isdir(folder_path):
                best_image, score = FolderProcessor.select_best_image_in_folder(folder_path)
                if best_image:
                    output_subfolder = os.path.join(output_folder, folder)
                    os.makedirs(output_subfolder, exist_ok=True)
                    output_path = os.path.join(output_subfolder, f"{folder}_best.jpg")
                    shutil.copy(best_image, output_path)
                    print(f"Best photo in '{folder}': {os.path.basename(best_image)}, score={score:.2f} -> {output_path}")
                else:
                    print(f"No suitable images found in '{folder}'")


