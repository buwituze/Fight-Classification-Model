# Fight Classification Project

This is a system that uses Machine Learning to detect fighting scenes in video content. This project has a pipeline that includes ML model (3D Convolutional Neural Network), FastAPI backend, and Next.js dashboard. The model was trained on video both fighting and normal (no-fight) videos.

## Links

- **Demo Video**: https://youtu.be/BD1qaQbf3eU
-
-

## Project Use Cases

- Security & Surveillance

  - CCTV Monitoring, Crowd Control,Prison Security

- Social Media Content Moderation

  - Filter violent content on platforms and apps

- Research & Analysis

  - Behavioral Studies, Sports Analysis (detect bad conduct in sports footage), Monitor patient behavior in facilities

## Key Features

### AI Model

- **3D CNN Architecture**: Optimized for temporal video analysis
- **Real-time Processing**: Processes 16 frames at 64x64 resolution
- **High Accuracy**: 91%+ accuracy with precision/recall optimization
- **Confidence Scoring**: Multi-level confidence analysis (High/Medium/Low)

### FastAPI Backend Endpoints

- **POST /health**: Check the health of the API.
- **POST /model**: Get information about the model, including its version and supported video formats.
- **POST /predict**: Make a prediction on a given video file.
- **POST /retrain**: Retrain the model with new data.
- **GET /training-status**: Get the status of the training process.
- **DELETE /cancel-training**: Cancel the training process.
- **GET /analytics/dataset-stats**: Get statistics about the dataset.
- **POST /predict** : Predict Fight
- **GET /analytics/confidence-analysis**: Get Confidence Analysis

### Next.js Dashboard

- **Integration**: connected with the FastAPI endpoints
- **Real-time Analytics**: Performance metrics and confidence analysis
- **Video Upload**: Drag-and-drop video prediction interface
- **Training Management**: Model retraining with progress visualization
- **Model Monitoring**: Health status and version history

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

## Project Setup

### Versions

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

### 2. Env setup

- Create the virtual env .venv: `python -m venv .venv`
- Activate the virtual env: `source .venv/bin/activate` or `source .venv/Scripts/activate` if you're using Linux or git bash
- Create `.env` file in the `frontend/` folder
- Place this variable in `.env`: NEXT_PUBLIC_API_BASE_URL=

### 3. Dataset Setup

- Unzip the UBI Fights dataset in the dataset folder
- For retraining, use the prepared trainin_data.zip dataset

##

### 4. Setup Backend

```bash
# Navigate to API directory
cd API

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server
python app.py
```

### 5. Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 6. Access Application

- **API Documentation**: http://127.0.0.1:8000/docs
- **Dashboard**: http://localhost:3000
- **Health Check**: http://127.0.0.1:8000/health

## API Usage

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

## Model Performance

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

## Configuration

### Supported Video Formats

- MP4, AVI, MOV, MKV, FLV, WMV, WebM

### File Size Limits

- **Single Video**: 100MB max
- **Training Data**: 500MB max

### Model Specifications

- **Input Shape**: (16, 64, 64, 3)
- **Architecture**: 3D CNN with BatchNorm and Dropout
- **Framework**: TensorFlow 2.x

## Training Data Format

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

- Minimum 50 videos per category (100 videos in total)
- Balanced distribution (40-60% fight ratio)
- Clear video quality recommended

## Development

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

## Flood Request usin Locust Results

### 10 users

### 50 Users

### 100 Users
