# Fight Classification Project

This is a system that uses Machine Learnin to detect fighting scenes in video content using a 3D Convolutional Neural Network. This project has an ML pipeline that includes ML model, FastAPI backend, and Next.js dashboard.

## Links

- **Demo Video**:
-
-

## Key Features

### AI Model

- **3D CNN Architecture**: Optimized for temporal video analysis
- **Real-time Processing**: Processes 16 frames at 64x64 resolution
- **High Accuracy**: 91%+ accuracy with precision/recall optimization
- **Confidence Scoring**: Multi-level confidence analysis (High/Medium/Low)

### FastAPI Backend

- **Async Processing**: Non-blocking video analysis
- **Auto-Retraining**: Upload new data to improve model performance
- **Progress Tracking**: Real-time training progress monitoring
- **Production Ready**: Health checks, error handling, and logging

### Next.js Dashboard

- **Real-time Analytics**: Performance metrics and confidence analysis
- **Video Upload**: Drag-and-drop video prediction interface
- **Training Management**: Model retraining with progress visualization
- **Model Monitoring**: Health status and version history

## Project Use Cases

### Security & Surveillance

- **CCTV Monitoring**: Automatically flag violent incidents in surveillance footage
- **Crowd Control**: Detect fights in public spaces, events, and venues
- **Prison Security**: Monitor inmate interactions for violence prevention

### Content Moderation

- **Social Media**: Filter violent content on platforms and apps
- **Streaming Services**: Classify and rate video content automatically
- **Educational Platforms**: Ensure safe content for students

### Research & Analysis

- **Behavioral Studies**: Analyze aggression patterns in research settings
- **Sports Analysis**: Detect unsportsmanlike conduct in sports footage
- **Healthcare**: Monitor patient behavior in psychiatric facilities

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Next.js UI   │ ←→ │   FastAPI      │ ←→ │   3D CNN Model  │
│   Dashboard     │    │   Backend      │    │   TensorFlow    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Video Upload  │    │ • Async Processing│   │ • Frame Extraction│
│ • Analytics     │    │ • Model Training │    │ • 3D Convolution │
│ • Training UI   │    │ • Health Monitoring│   │ • Classification │
│ • Status Monitor│    │ • Data Validation │    │ • Confidence Score│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## What To Have

### Prerequisites

```bash
# Python 3.8+
python --version

# Node.js 16+
node --version
```

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/fight-detection-system.git
cd Fight-Classification-Model
```

### 2. Setup Backend

```bash
# Navigate to API directory
cd API

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
python app.py
```

### 3. Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 4. Access Application

- **API Documentation**: http://127.0.0.1:8000/docs
- **Dashboard**: http://localhost:3000
- **Health Check**: http://127.0.0.1:8000/health

## 📝 API Usage

### Predict Video Content

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@video.mp4" \
  -F "threshold=0.5"
```

**Response:**

```json
{
  "prediction": "fight",
  "probability": 0.87,
  "confidence_score": 0.87,
  "confidence_level": "High",
  "frames_processed": 16
}
```

### Retrain Model

```bash
curl -X POST "http://127.0.0.1:8000/retrain" \
  -F "training_data=@training_data.zip" \
  -F "epochs=20" \
  -F "batch_size=4"
```

### Check Training Status

```bash
curl http://127.0.0.1:8000/training-status
```

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 91.2% |
| Precision | 87.4% |
| Recall    | 71.3% |
| F1-Score  | 78.6% |

### Confidence Distribution

- **High Confidence (>80%)**: 75% of predictions
- **Medium Confidence (60-80%)**: 20% of predictions
- **Low Confidence (<60%)**: 5% of predictions

## 🔧 Configuration

### Supported Video Formats

- MP4, AVI, MOV, MKV, FLV, WMV, WebM

### File Size Limits

- **Single Video**: 100MB max
- **Training Data**: 500MB max

### Model Specifications

- **Input Shape**: (16, 64, 64, 3)
- **Architecture**: 3D CNN with BatchNorm and Dropout
- **Framework**: TensorFlow 2.x

## 📁 Training Data Format

Prepare your training data as a ZIP file:

```
training_data.zip
├── normal/          # Non-fighting videos
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
└── fight/           # Fighting videos
    ├── fight1.mp4
    ├── fight2.avi
    └── ...
```

**Requirements:**

- Minimum 10 videos per category
- Balanced distribution (20-80% fight ratio)
- Clear video quality recommended

## 🛠️ Development

### Project Structure

```
fight-detection-system/
├── Notebook/                      # Jupyter notebook for model development
│   ├── README.md
│   ├── Fihting_3D_CNN_Model.ipynb
├── API/                           # FastAPI backend
│   ├── fight_detection_model_optimized.h5
│   ├── README.md
│   ├── requirements.txt
│   ├── app.py
│   ├── Dockerfile
├── frontend/                       # Next.js dashboard
│   ├── src/, env, packae.json e.t.c
│   ├── Dockerfile
│   └── README.md
├── Model/                          # Holds the saved trained model
│   ├── fight_detection_model_optimized.h5
├── dataset/                        # Holds zipped dataset folder
│   ├── training_data.zip
├── locustfile.py                   # Run this file for locust tests
└── README.md
```

### Key Technologies

- **Backend**: FastAPI, TensorFlow, OpenCV, scikit-learn
- **Frontend**: Next.js, Tailwind CSS, Chart.js
- **Model**: 3D CNN, Keras, NumPy

## Monitoring & Analytics

The dashboard provides comprehensive monitoring:

- **Real-time Training Progress**: Live epoch updates and loss tracking
- **Model Performance History**: Accuracy trends over time
- **Confidence Analysis**: Prediction reliability metrics
- **System Health**: API status and model availability
