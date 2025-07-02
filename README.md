# 🏎️ Pakistani License Plate Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue) ![YOLO](https://img.shields.io/badge/YOLO-v8-orange) ![OpenCV](https://img.shields.io/badge/OpenCV-4.10-green) ![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.9.1-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

A computer vision system designed to detect and extract text from Pakistani license plates using YOLOv8 for object detection and PaddleOCR for text recognition.

## 🌟 Features
- **License Plate Detection**: Detects Pakistani license plates in images using a fine-tuned YOLOv8 model.
- **Text Extraction**: Extracts alphanumeric text from detected license plates using PaddleOCR.
- **Image Processing**:
  - Processes multiple images in a directory.
  - Draws bounding boxes around detected license plates with extracted text annotations.
  - Saves annotated images in an output directory.
- **Performance Optimized**:
  - GPU acceleration support (NVIDIA Tesla T4 compatible).
  - Configurable image size and batch processing for efficiency.

## 🛠️ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/pakistani-license-plate-detection.git
   cd pakistani-license-plate-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install ultralytics<=8.3.78 opencv-python paddleocr paddlepaddle
   ```

3. **Download model weights**:
   - Place the trained YOLOv8 model weights (`best.pt`) in the project directory. You can use the pre-trained model from the dataset or train your own (see [Training](#⚙️-training)).

## 🚀 Usage
Run the license plate detection and text extraction on a folder of images:

```bash
python detect_license_plates.py --input /path/to/image_folder --output /path/to/output_folder
```

### Optional Arguments
- `--model`: Path to custom YOLOv8 weights (default: `best.pt`).
- `--input`: Path to the folder containing input images.
- `--output`: Path to save annotated images (default: `./license_plate_output/`).

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

# Detect license plates in an image folder
results = model.predict(source="path/to/images", save=False, classes=22)

# Process each image
for i, result in enumerate(results):
    image = result.orig_img
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = image[y1:y2, x1:x2]
        gray_plate = cv2.cvtColor(plate_crop, cv12.COLOR_BGR2GRAY)
        ocr_result = ocr.ocr(gray_plate, cls=True)
        plate_text = "".join([word[1][0] for line in ocr_result for word in line if line])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        cv2.putText(image, plate_text, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imwrite(f"output/result_{i}.png", image)
```

## 📂 Project Structure
```
pakistani-license-plate-detection/
├── detect_license_plates.py   # Main script for detection and OCR
├── best.pt                   # YOLOv8 model weights
├── data.yaml                 # Dataset configuration for training
├── input_images/            # Folder for input images
├── output/                  # Processed results (annotated images)
└── README.md                # This documentation
```

## 🔍 Detection Classes
The YOLOv8 model is trained to detect the following classes:

| Class ID | Class Name       | Description                     |
|----------|------------------|---------------------------------|
| 0-9      | 0-9              | Digits on license plates        |
| 10-36    | A-Z              | Letters on license plates       |
| 22       | LicensePlate     |禁止

## ⚙️ Technical Details
### Processing Pipeline
1. **Image Loading**: Loads images from the specified input folder.
2. **YOLOv8 Inference**: Detects license plates (class ID 22) in each image.
3. **Region Cropping**: Extracts the license plate region from the image.
4. **Grayscale Conversion**: Converts the cropped region to grayscale for better OCR accuracy.
5. **PaddleOCR Processing**: Extracts alphanumeric text from the license plate region.
6. **Annotation**: Draws green bounding boxes and text annotations on the original image.
7. **Output Saving**: Saves annotated images to the output directory.

### Performance
- **Inference Speed**: ~12-36ms per image on NVIDIA Tesla T4 GPU.
- **Environment**: Python 3.10.12, Ultralytics 8.3.78, PaddleOCR 2.9.1, OpenCV 4.10.0.
- **Hardware**: Optimized for GPU acceleration, tested on NVIDIA Tesla T4 with 15095MiB VRAM.

## ⚙️ Training
To train your own YOLOv8 model for license plate detection:
1. Prepare a dataset with labeled Pakistani license plates (e.g., from Kaggle dataset: [Pakistani License Plate Dataset](https://www.kaggle.com/datasets/6690813/pakistani-license-pate-dataset)).
2. Configure the dataset in `data.yaml`.
3. Run the training command:
   ```bash
   yolo train model=yolov8m.pt data=/path/to/data.yaml epochs=50 imgsz=640 batch=8
   ```
4. Save the trained weights (`best.pt`) for use in detection.

## 📈 Future Improvements
- **Real-time Processing**: Integrate webcam or video stream for live detection.
- **Improved OCR Accuracy**: Fine-tune PaddleOCR for Pakistani license plate formats.
- **Multi-plate Detection**: Handle multiple license plates in a single image.
- **Web Interface**: Develop a web-based dashboard for viewing results.
- **Database Integration**: Store detected plate numbers in a database for tracking.

## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-detection`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-detection`).
5. Open a Pull Request.

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---
