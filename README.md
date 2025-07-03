🏎️ Pakistani License Plate Recognition System
    
A computer vision system developed at the NCAI (National Center for Artificial Intelligence) Lab to recognize Pakistani license plates in images and videos by detecting plates and extracting text using YOLOv8 for object detection and PaddleOCR for text recognition. Special thanks to my supervisor, Umar Sadique, for his guidance and support in this project.
🌟 Features

License Plate Recognition: Detects and extracts text from Pakistani license plates in images and videos using a fine-tuned YOLOv8 model and PaddleOCR.
Text Extraction: Accurately reads alphanumeric text from detected license plates.
Image and Video Processing:
Processes multiple images in a directory or frames from video files/webcams.
Draws green bounding boxes with text annotations on detected plates.
Saves annotated images or compiles annotated frames into an output video.


Performance Optimized:
GPU acceleration support (NVIDIA Tesla T4 compatible).
Configurable image size and batch processing for efficiency.



🛠️ Installation

Clone the repository:
git clone https://github.com/yourusername/pakistani-license-plate-recognition.git
cd pakistani-license-plate-recognition


Install dependencies:
pip install ultralytics<=8.3.78 opencv-python paddleocr paddlepaddle


Download model weights:

Place the trained YOLOv8 model weights (best.pt) in the project directory. Use the pre-trained model or train your own (see Training).



🚀 Usage
Process Images
Run license plate recognition on a folder of images:
python recognize_license_plates.py --input /path/to/image_folder --output /path/to/output_folder

Process Videos
Run license plate recognition on a video file or webcam:
python process_video.py --input /path/to/video.mp4 --output output_video.mp4

For webcam input:
python process_video.py --input 0 --output output_video.mp4

Optional Arguments

--model: Path to custom YOLOv8 weights (default: best.pt).
--input: Path to image folder, video file, or 0 for webcam.
--output: Path to save annotated images or video (default: ./license_plate_output/ or output_video.mp4).

Example Code (Video Processing)
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2

# Load YOLO model
model = YOLO("best.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Open video
cap = cv2.VideoCapture("input_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, save=False, classes=22)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            ocr_result = ocr.ocr(gray_plate, cls=True)
            plate_text = "".join([word[1][0] for line in ocr_result for word in line if line])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
            cv2.putText(frame, plate_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()

📂 Project Structure
pakistani-license-plate-recognition/
├── recognize_license_plates.py   # Script for image-based recognition
├── process_video.py             # Script for video-based recognition
├── best.pt                     # YOLOv8 model weights
├── data.yaml                   # Dataset configuration for training
├── input_images/               # Folder for input images
├── input_videos/               # Folder for input videos
├── output/                     # Processed results (images/videos)
└── README.md                   # This documentation

🔍 Recognition Classes



Class ID
Class Name
Description



0-9
0-9
Digits on license plates


10-36
A-Z
Letters on license plates


22
LicensePlate
Entire license plate region


⚙️ Technical Details
Recognition Pipeline

Input Loading: Loads images or video frames (from file or webcam).
YOLOv8 Inference: Detects license plates (class ID 22) in each frame/image.
Region Cropping: Extracts the license plate region.
Grayscale Conversion: Converts cropped region to grayscale for improved OCR accuracy.
PaddleOCR Processing: Extracts alphanumeric text from the license plate.
Annotation: Draws green bounding boxes and text on the frame/image.
Output Saving: Saves annotated images or compiles frames into an output video.

Performance

Inference Speed: ~12-36ms per frame/image on NVIDIA Tesla T4 GPU.
Environment: Python 3.10.12, Ultralytics 8.3.78, PaddleOCR 2.9.1, OpenCV 4.10.0.
Hardware: Optimized for GPU acceleration, tested on NVIDIA Tesla T4 (15095MiB VRAM).
Video Processing: Real-time capable at ~30 FPS for standard resolutions.

⚙️ Training
To train your own YOLOv8 model for license plate recognition:

Prepare a dataset (e.g., Pakistani License Plate Dataset).
Configure the dataset in data.yaml.
Run:yolo train model=yolov8m.pt data=/path/to/data.yaml epochs=50 imgsz=640 batch=8



📈 Future Improvements

Real-time Optimization: Further reduce latency for live webcam recognition.
Enhanced OCR Accuracy: Fine-tune PaddleOCR for diverse Pakistani plate formats.
Multi-plate Recognition: Handle multiple plates in a single frame.
Web Interface: Develop a dashboard for viewing results.
Database Integration: Store recognized plate numbers for tracking.

🤝 Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/recognition-enhancement).
Commit changes (git commit -m 'Add recognition feature').
Push to the branch (git push origin feature/recognition-enhancement).
Open a Pull Request.

📜 License
Distributed under the MIT License. See LICENSE for more information.
🙏 Acknowledgments

Umar Sadique, my supervisor, for his invaluable guidance and mentorship.
NCAI Lab for providing an innovative environment and resources to develop this project.
