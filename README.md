# 🏎️ Pakistani License Plate Recognition System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![YOLO](https://img.shields.io/badge/YOLO-v8-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green) ![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.9.1-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

A computer vision system developed at the **NCAI (National Center for Artificial Intelligence) Lab** to recognize Pakistani license plates in images and videos by detecting plates and extracting text using YOLOv8 for object detection and PaddleOCR for text recognition. Special thanks to my supervisor, **Umar Sadique**, for his invaluable guidance and support.

## 🌟 Features
- **License Plate Recognition**: Detects and extracts alphanumeric text from Pakistani license plates using a fine-tuned YOLOv8 model and PaddleOCR.
- **Image and Video Processing**:
  - Processes multiple images in a directory or frames from video files/webcams.
  - Draws green bounding boxes with text annotations on detected plates.
  - Saves annotated images or compiles annotated frames into an output video.
- **Performance Optimized**:
  - GPU acceleration support (NVIDIA Tesla T4 compatible).
  - Configurable image size and batch processing for efficiency.

## 🛠️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pakistani-license-plate-recognition.git
   cd pakistani-license-plate-recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install ultralytics<=8.3.78 opencv-python paddleocr paddlepaddle
   ```

3. **Download model weights**:
   - Place the trained YOLOv8 model weights (`best.pt`) in the project directory. Use the pre-trained model from the dataset or train your own (see [Training](#⚙️-training)).

## 🚀 Usage
Run license plate recognition on a folder of images or a video:

```bash
python recognize_license_plates.py --input /path/to/image_folder --output /path/to/output_folder
```

For video or webcam input:
```bash
python process_video.py --input /path/to/video.mp4 --output /path/to/output_video.mp4
```

For webcam:
```bash
python process_video.py --input 0 --output /path/to/output_video.mp4
```

### Optional Arguments
- `--model`: Path to custom YOLOv8 weights (default: `best.pt`).
- `--input`: Path to image folder, video file, or `0` for webcam.
- `--output`: Path to save annotated images or video (default: `./license_plate_output/` or `output_video.mp4`).

### Example Code
```python
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import os

# Load YOLO model
model = YOLO("best.pt")

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Process video
cap = cv2.VideoCapture("input_video.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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
```

## 📂 Project Structure
```
pakistani-license-plate-recognition/
├── recognize_license_plates.py   # Main script for image-based recognition
├── process_video.py             # Script for video-based recognition
├── best.pt                     # YOLOv8 model weights
├── data.yaml                   # Dataset configuration for training
├── input_images/               # Folder for input images
├── input_videos/               # Folder for input videos
├── output/                     # Processed results (images/videos)
└── README.md                   # This documentation
```

## 🔍 Recognition Classes
The YOLOv8 model is trained to detect the following classes:

| Class ID | Class Name       | Description                     |
|----------|------------------|---------------------------------|
| 0-9      | 0-9              | Digits on license plates        |
| 10-36    | A-Z              | Letters on license plates       |
| 22       | LicensePlate     | Entire license plate region     |

## ⚙️ Technical Details
### Recognition Pipeline
1. **Image/Video Loading**: Loads images from a folder or frames from a video file/webcam.
2. **YOLOv8 Inference**: Detects license plates (class ID 22) in each image/frame.
3. **Region Cropping**: Extracts the license plate region from the image/frame.
4. **Grayscale Conversion**: Converts the cropped region to grayscale for better OCR accuracy.
5. **PaddleOCR Processing**: Extracts alphanumeric text from the license plate region.
6. **Annotation**: Draws green bounding boxes and text annotations on the image/frame.
7. **Output Saving**: Saves annotated images or compiles annotated frames into an output video.

### Performance
- **Inference Speed**: ~12-36ms per image/frame on NVIDIA Tesla T4 GPU.
- **Environment**: Python 3.10.12, Ultralytics 8.3.78, PaddleOCR 2.9.1, OpenCV 4.10.0.
- **Hardware**: Optimized for GPU acceleration, tested on NVIDIA Tesla T4 with 15095MiB VRAM.
- **Video Processing**: Real-time capable at ~30 FPS for standard resolutions.

## ⚙️ Training
To train your own YOLOv8 model for license plate recognition:
1. Prepare a dataset with labeled Pakistani license plates (e.g., from Kaggle dataset: [Pakistani License Plate Dataset](https://www.kaggle.com/datasets/6690813/pakistani-license-pate-dataset)).
2. Configure the dataset in `data.yaml`.
3. Run the training command:
   ```bash
   yolo train model=yolov8m.pt data=/path/to/data.yaml epochs=50 imgsz=640 batch=8
   ```
4. Save the trained weights (`best.pt`) for use in recognition.

## 📈 Future Improvements
- **Real-time Processing**: Optimize for live webcam recognition with minimal latency.
- **Improved OCR Accuracy**: Fine-tune PaddleOCR for Pakistani license plate formats.
- **Multi-plate Recognition**: Handle multiple license plates in a single image/frame.
- **Web Interface**: Develop a web-based dashboard for viewing results.
- **Database Integration**: Store recognized plate numbers in a database for tracking.

## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-recognition`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-recognition`).
5. Open a Pull Request.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments
- **Umar Sadique**, my supervisor, for his invaluable guidance and mentorship.
- **NCAI Lab** for providing an innovative environment and resources to develop this project.

---
