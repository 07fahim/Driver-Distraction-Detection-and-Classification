# SafeDrive AI - Distracted Driver Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/yeager07/distracted-driving-detection)

AI-powered real-time driver safety monitoring system using advanced computer vision to detect and classify 12 types of distracted driving behaviors.

---

## Table of Contents

- [Overview](#overview)
- [Live Demos](#live-demos)
- [Features](#features)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Overview

SafeDrive AI is an advanced computer vision system designed to detect and prevent driver distractions in real-time. Using state-of-the-art YOLO object detection and deep learning classification models, the system identifies 12 different types of dangerous driving behaviors with high accuracy.

### Detected Behaviors (12 Classes)

**Phone Usage**
- c1: Texting
- c2: Talking on the phone

**Vehicle Controls**
- c3: Operating the radio
- c0: Safe driving

**Physical Activities**
- c4: Drinking
- c5: Reaching behind
- c6: Hair and makeup

**Social Interaction**
- c7: Talking to passenger

**Drowsiness Detection**
- d0: Eyes closed
- d1: Yawning
- d2: Nodding off
- d3: Eyes open

---

## Live Demos

### Gradio Interface
**Platform:** Hugging Face Spaces  
**Technology:** Gradio + ONNX Runtime  
**Link:** [https://huggingface.co/spaces/yeager07/distracted-driving-detection](https://huggingface.co/spaces/yeager07/distracted-driving-detection)

### Flask Web Application
**Platform:** Render.com  
**Technology:** Flask + Docker + HF Spaces API  
**Link:** [Insert Your Render URL Here]

---

## Features

### Detection Capabilities
- 12-class behavioral detection with 98.1% mAP@50
- Real-time processing for images and videos
- High accuracy with optimized inference speed
- Multiple model backends (ONNX Runtime)

### Web Interfaces
- **Gradio:** Interactive drag-and-drop interface
- **Flask:** Modern responsive web application
- Analytics dashboards with detailed statistics
- Video processing with frame-by-frame analysis

### Deployment Options
- Docker containerization for consistent environments
- CI/CD pipeline via GitHub Actions
- Cloud hosting on Render.com and Hugging Face Spaces
- RESTful API for easy integration

---

## Dataset

### Statistics

| Split | Images | Percentage |
|-------|--------|-----------|
| Training | 17,909 | 80% |
| Validation | 2,238 | 10% |
| Testing | 2,240 | 10% |
| **Total** | **22,390** | **100%** |

**Source:** [Roboflow Universe - Distracted Driving Dataset v2](https://universe.roboflow.com/flytech/distracted-driving-v2wk5-f5vtj/dataset/1)

### Characteristics
- High-quality annotations for all 12 behavior classes
- Diverse lighting conditions and camera angles
- Multiple driver demographics and vehicle types
- Rigorous train/validation/test split
- Publicly available for research

---

## Model Performance

### Detection Models (YOLO)

| Model | GFLOPs | Inference (ms) | mAP@50 | mAP@50-95 | Precision | Recall |
|-------|--------|----------------|--------|-----------|-----------|--------|
| YOLOv10N | 8.2 | 3.6 | 0.978 | 0.845 | 0.968 | 0.959 |
| YOLOv10S | 24.5 | 8.5 | **0.981** | 0.856 | 0.959 | 0.977 |
| YOLOv10M | 63.5 | 19.5 | 0.981 | 0.854 | 0.963 | 0.971 |
| YOLOv10L | 126.4 | 31.2 | 0.981 | 0.854 | 0.966 | 0.979 |

**Selected Model:** YOLOv10S (optimal balance of speed and accuracy)

### Classification Models

| Model | Accuracy | AUC (OVR) | Inference/Batch (s) | Inference/Sample (s) |
|-------|----------|-----------|---------------------|---------------------|
| ConvNeXt-Tiny | **97.67%** | **0.9988** | 0.0091 | 0.000285 |
| VGG19 | 96.98% | 0.9976 | **0.0028** | **0.000086** |
| ResNet50 | 95.88% | 0.9957 | 0.0107 | 0.000334 |
| EfficientNet-B0 | 95.88% | 0.9954 | 0.0133 | 0.000416 |
| MobileNetV2 | 95.47% | 0.9974 | 0.0082 | 0.000255 |

**Key Insights:**
- ConvNeXt-Tiny achieves highest accuracy and AUC
- VGG19 offers fastest inference time
- MobileNetV2 optimized for mobile deployment

---

## Architecture

### System Flow

```
User Interface Layer
├── Gradio App (HF Spaces)
└── Flask App (Render.com)
        ↓
Backend Services
├── ONNX Runtime (Local Inference)
└── Gradio Client API (HF Spaces)
        ↓
AI Model Layer
└── YOLOv10S Detection Model (best.onnx)
    ├── 12 behavior classes
    ├── 640×640 input size
    └── 98.1% mAP@50 accuracy
```

### Processing Pipeline

1. **Upload** - User uploads image/video
2. **Preprocessing** - Resize to 640×640, normalize, convert to tensor
3. **Inference** - YOLO ONNX model detects behaviors
4. **Post-processing** - Parse predictions, filter by confidence threshold
5. **Visualization** - Overlay text labels with confidence scores
6. **Response** - Return annotated media with statistics

---

## Installation

### Gradio Application

**Prerequisites:**
- Python 3.12+
- ONNX Runtime

**Setup:**

```bash
# Clone repository
git clone https://github.com/yourusername/safedrive-ai.git
cd safedrive-ai

# Switch to main branch
git checkout main

# Navigate to deployment folder
cd deployment

# Install dependencies
pip install -r requirements.txt

# Run Gradio app
python app.py
```

Access at: `http://localhost:7860`

### Flask Web Application

**Prerequisites:**
- Python 3.12+
- Docker (optional)

**Local Setup:**

```bash
# Clone repository
git clone https://github.com/yourusername/safedrive-ai.git
cd safedrive-ai

# Switch to flask branch
git checkout flask

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

Access at: `http://localhost:5000`

**Docker Setup:**

```bash
# Build image
docker build -t safedrive-ai .

# Run container
docker run -p 5000:5000 safedrive-ai
```

---

## Usage

### Gradio Interface

Access the interactive Gradio interface at [https://huggingface.co/spaces/yeager07/distracted-driving-detection](https://huggingface.co/spaces/yeager07/distracted-driving-detection)

**Features:**
- Upload images or videos via drag-and-drop
- Real-time detection with confidence scores
- Interactive results visualization
- Detection summary tables
- Webcam support for live detection

### Flask Web Application

Access the Flask web interface at your deployed URL.

**Available Pages:**
- **Home** - Overview and features
- **Image Detection** - Upload and analyze images
- **Video Detection** - Process video files
- **About** - Technical details and documentation

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect-image` | POST | Upload image for detection |
| `/api/detect-video` | POST | Upload video for processing |
| `/api/download-video/<id>` | GET | Download processed video |

**Response Format:**

Image detection returns JSON with annotated image (base64) and detection list including class names and confidence scores.

Video detection returns JSON with video ID and summary statistics including total detections, average confidence, frame ranges, and duration per behavior.

---

## Deployment

### Gradio (Hugging Face Spaces)

**Platform:** Hugging Face Spaces  
**Repository:** `main` branch, `deployment/` folder  
**Runtime:** Python 3.12 with ONNX Runtime  
**Live URL:** [https://huggingface.co/spaces/yeager07/distracted-driving-detection](https://huggingface.co/spaces/yeager07/distracted-driving-detection)

**Application Demo:**

https://github.com/user-attachments/assets/your-gradio-video-id.mp4

*Interactive Gradio interface demonstration with image and video detection capabilities*

**Deployment Process:**
1. Code pushed to `main` branch triggers automatic rebuild
2. Hugging Face Spaces pulls latest changes
3. Dependencies installed from requirements.txt
4. ONNX model weights loaded from repository
5. Gradio application starts on port 7860
6. Public URL becomes accessible

**Configuration:**
- Automatic rebuilds on push
- Model inference via ONNX Runtime CPU
- Public access with no authentication required

### Flask (Render + Docker)

**Platform:** Render.com  
**Repository:** `flask` branch  
**Container:** Docker with Python 3.12-slim base  
**Web Server:** Gunicorn with 2 workers, 300-second timeout  
**CI/CD:** GitHub Actions automated deployment  
**Live URL:** [Insert Your Render URL Here]

**Application Demo:**

https://github.com/user-attachments/assets/your-flask-video-id.mp4

*Complete Flask web application walkthrough showing all features and detection capabilities*

**Deployment Process:**
1. Code pushed to `flask` branch
2. GitHub Actions workflow triggered
3. Render API called with deploy hook
4. Docker image built from Dockerfile
5. Application deployed with Gunicorn
6. Health checks verify successful deployment

**Infrastructure:**
- Docker containerization for consistency
- Gunicorn WSGI server for production
- Automatic file cleanup after 1 hour
- 100MB maximum file upload size
- /tmp directories for file processing

**GitHub Actions Workflow:**

Automated deployment configured in `.github/workflows/deploy.yml` with triggers on push to flask branch, Render API integration, and automatic cache clearing.

**Environment Configuration:**

Required secrets in GitHub repository settings:
- `RENDER_API_KEY` - Render API authentication
- `RENDER_SERVICE_ID` - Target service identifier

---

## Repository Structure

```
safedrive-ai/
│
├── main/                          # Main branch
│   ├── notebooks/                 # Model training
│   │   ├── detection/
│   │   │   ├── yolov10n.ipynb
│   │   │   ├── yolov10s.ipynb
│   │   │   ├── yolov10m.ipynb
│   │   │   └── yolov10l.ipynb
│   │   └── classification/
│   │       ├── convnext-tiny.ipynb
│   │       ├── efficientnet-b0.ipynb
│   │       ├── mobilenetv2.ipynb
│   │       ├── resnet50.ipynb
│   │       └── vgg19.ipynb
│   └── deployment/                # Gradio app
│       ├── app.py
│       ├── best.onnx
│       └── requirements.txt
│
├── flask/                         # Flask branch
│   ├── app.py                     # Flask application
│   ├── templates/                 # HTML templates
│   │   ├── index.html
│   │   ├── image.html
│   │   ├── video.html
│   │   └── about.html
│   ├── static/css/                # Stylesheets
│   │   ├── style.css
│   │   ├── home.css
│   │   ├── detection.css
│   │   └── analysis.css
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── gunicorn.conf.py
│   └── .github/workflows/
│       └── deploy.yml
│
└── README.md
```

---

## Technologies

**Deep Learning & AI**
- YOLO (You Only Look Once) - Object detection
- ONNX Runtime - Cross-platform inference
- OpenCV - Computer vision
- NumPy - Numerical computing

**Web Frameworks**
- Flask 3.0.0 - Backend framework
- Gradio - ML interface framework
- Gunicorn - WSGI HTTP server

**Frontend**
- HTML5/CSS3 - Modern web standards
- JavaScript - Interactive features
- Font Awesome - Icon library

**DevOps & Deployment**
- Docker - Containerization
- GitHub Actions - CI/CD automation
- Render.com - Cloud hosting
- Hugging Face Spaces - ML deployment

**Development Tools**
- Python 3.12 - Programming language
- Git - Version control
- Jupyter Notebooks - Experimentation

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure CI/CD pipeline passes

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Project Maintainer:** Your Name

- Email: contact@safedrive-ai.com
- Website: [safedrive-ai.com](#)
- LinkedIn: [Your LinkedIn](#)
- GitHub: [@yourusername](https://github.com/yourusername)

**Project Links:**
- Gradio App: [https://huggingface.co/spaces/yeager07/distracted-driving-detection](https://huggingface.co/spaces/yeager07/distracted-driving-detection)
- Flask App: [Your Render URL](#)
- Dataset: [Roboflow Universe](https://universe.roboflow.com/flytech/distracted-driving-v2wk5-f5vtj/dataset/1)

---

## Acknowledgments

- Dataset provided by FlyTech team on Roboflow Universe
- YOLO models from Ultralytics YOLOv10
- Deployment platforms: Hugging Face Spaces and Render.com
- Open-source community for tools and libraries

---

**Last Updated:** January 2025

**Status:** Production Ready

---

**SafeDrive AI - Drive Safe, Save Lives**

Star this repository if you find it helpful.
