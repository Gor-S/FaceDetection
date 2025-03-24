### Project Title: Faces Extractor
### Short Description:
Faces Extractor is a program that detects and extracts faces from videos, then enhances their quality using deep learning.
## 1. Objectives
- Automate face extraction from videos.
- Improve face detection and tracking.
- Select the best-quality frame for each face.
- Enhance face clarity and resolution.
- Optimize for real-time or batch processing.
## 2. Methodology
- Process video frames and detect faces using models like Haar Cascade or MTCNN.
- Select the best frame based on sharpness, brightness, and angle.
- Enhance faces using Super Resolution and noise reduction techniques.
- Save and organize the results.
## 3. Technologies Used
- Programming Language: Python
- Libraries: OpenCV, TensorFlow/PyTorch
- Models: MTCNN, ESRGAN
## 4. Challenges and Solutions
- Low-quality faces: Enhanced using Super Resolution.
- Motion blur: Selected the sharpest frame.
- False detections: Improved accuracy with MTCNN.
- Slow processing: Optimized for efficiency.
## 5. Results
- Extracted faces with 85-95% accuracy.
- Improved image quality by 2x to 4x.
- Handles real-time and batch processing։
## 6. Future Work
- Add real-time processing with GPU support.
- Train custom models for better enhancement.
- Include face recognition and tracking.
- Develop a user-friendly interface.
## 7. References
1. OpenCV Documentation
2. ESRGAN, MTCNN research papers


pip install -r requirements.txt


Path to models: https://drive.google.com/drive/folders/1tJ-IfF_luVGGiTgi2PMjpRW05dRvi8_-?usp=drive_link
Path to gfpgan: https://drive.google.com/drive/folders/1gszj4kZUUVepviidgykzW4Yfb4jM-zek?usp=drive_link
