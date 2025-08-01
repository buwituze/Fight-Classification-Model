# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
import cv2
import uvicorn
import os
import tempfile
from typing import Optional, List, Dict, Any
import logging
import asyncio
from contextlib import asynccontextmanager
import json
import zipfile
import shutil
from datetime import datetime
import uuid
import threading
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG_SIZE = 64
FRAMES_PER_VIDEO = 16

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "fight_detection_model_optimized.h5")
RETRAIN_DATA_DIR = os.path.join(CURRENT_DIR, "retrain_data")
TEMP_MODEL_PATH = os.path.join(CURRENT_DIR, "temp_retrained_model.h5")

# Ensure retrain data directory exists
os.makedirs(RETRAIN_DATA_DIR, exist_ok=True)

model = None

# Global training status
training_status = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "started_at": None,
    "completed_at": None,
    "error": None,
    "task_id": None,
    "epochs_completed": 0,
    "total_epochs": 0,
    "current_loss": None,
    "current_accuracy": None,
    "videos_processed": 0,
    "total_videos": 0
}

training_lock = threading.Lock()

def load_model_with_custom_objects():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        logger.warning(f"Standard loading failed: {e}")
        
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("Model loaded without compilation")
            return model
        except Exception as e2:
            logger.warning(f"Loading without compilation failed: {e2}")
            
            try:
                logger.error("Could not load model. You may need to recreate the model architecture.")
                return None
            except Exception as e3:
                logger.error(f"All loading methods failed: {e3}")
                return None

def create_fallback_model():
    logger.info("Creating fallback model...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D((2, 2, 2)),
        tf.keras.layers.GlobalAveragePooling3D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    logger.warning("Using fallback model - predictions will be random!")
    return model

def create_enhanced_model():
    """Create the enhanced 3D CNN model architecture"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv3D, MaxPooling3D, Dense, Dropout, 
        BatchNormalization, Input, GlobalAveragePooling3D
    )
    from tensorflow.keras.regularizers import l2
    
    model = Sequential()
    model.add(Input(shape=(FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE, 3)))
    
    # First Conv3D block
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(0.3))
    
    # Second Conv3D block
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.4))
    
    # Third Conv3D block
    model.add(Conv3D(96, kernel_size=(3, 3, 3), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.4))
    
    # Dense layers
    model.add(GlobalAveragePooling3D())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def augment_frames(frames):
    """Data augmentation for video frames"""
    augmented = frames.copy()
    
    # Horizontal flip
    if random.random() > 0.5:
        augmented = np.flip(augmented, axis=2)
    
    # Brightness adjustment
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness_factor, 0, 255)
    
    # Rotation
    if random.random() > 0.7:
        angle = random.uniform(-5, 5)
        for i in range(len(augmented)):
            center = (IMG_SIZE // 2, IMG_SIZE // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented[i] = cv2.warpAffine(augmented[i], rotation_matrix, (IMG_SIZE, IMG_SIZE))
    
    return augmented.astype(np.uint8)

def extract_frames_for_training(video_path, max_frames=FRAMES_PER_VIDEO):
    """Extract frames for training (similar to your original function)"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < max_frames:
        step = 1
    else:
        step = total_frames // max_frames
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
        
        if len(frames) == max_frames:
            break
    
    cap.release()
    
    # Pad with last frame if necessary
    while len(frames) < max_frames:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    
    return np.array(frames[:max_frames])

def process_training_data(data_dir: str, use_augmentation: bool = True):
    """Process uploaded training data"""
    global training_status
    
    X, y = [], []
    folder_mapping = {'normal': 0, 'fight': 1}
    
    total_videos = 0
    for folder in folder_mapping.keys():
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            video_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
            total_videos += len(video_files)
    
    training_status["total_videos"] = total_videos
    processed_videos = 0
    
    for folder, label in folder_mapping.items():
        folder_path = os.path.join(data_dir, folder)
        
        if not os.path.exists(folder_path):
            logger.warning(f"Folder {folder_path} not found")
            continue
            
        video_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))]
        
        logger.info(f"Processing {len(video_files)} videos from '{folder}' folder...")
        
        for i, video_file in enumerate(video_files):
            try:
                video_path = os.path.join(folder_path, video_file)
                frames = extract_frames_for_training(video_path)
                
                X.append(frames)
                y.append(label)
                
                # Data augmentation
                if use_augmentation and random.random() > 0.7:
                    augmented_frames = augment_frames(frames)
                    X.append(augmented_frames)
                    y.append(label)
                
                processed_videos += 1
                training_status["videos_processed"] = processed_videos
                
                # Update progress
                progress = int((processed_videos / total_videos) * 20)  # 20% for data processing
                training_status["progress"] = progress
                
                if i % 10 == 0:
                    logger.info(f"  Processed {i+1}/{len(video_files)} videos from {folder}")
                    
            except Exception as e:
                logger.warning(f"Error processing {video_file}: {str(e)}")
                continue
    
    if len(X) == 0:
        raise ValueError("No valid video data found")
    
    X = np.array(X).astype(np.float32) / 255.0
    y = np.array(y)
    
    # Shuffle data
    X, y = shuffle(X, y, random_state=42)
    
    logger.info(f"Processed data shape: {X.shape}")
    logger.info(f"Class distribution: {Counter(y)}")
    
    return X, y

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to update training progress"""
    
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
    
    def on_epoch_end(self, epoch, logs=None):
        global training_status
        
        with training_lock:
            training_status["epochs_completed"] = epoch + 1
            training_status["current_loss"] = float(logs.get('loss', 0))
            training_status["current_accuracy"] = float(logs.get('accuracy', 0))
            
            # Progress: 20% for data processing + 70% for training + 10% for saving
            epoch_progress = int(((epoch + 1) / self.total_epochs) * 70)
            training_status["progress"] = 20 + epoch_progress
            
            logger.info(f"Epoch {epoch + 1}/{self.total_epochs} - "
                       f"Loss: {logs.get('loss', 0):.4f}, "
                       f"Accuracy: {logs.get('accuracy', 0):.4f}")

def retrain_model_background(data_dir: str, epochs: int, batch_size: int, validation_split: float, task_id: str):
    """Background task for model retraining"""
    global model, training_status
    
    try:
        with training_lock:
            training_status.update({
                "is_training": True,
                "progress": 0,
                "status": "preparing_data",
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "error": None,
                "task_id": task_id,
                "epochs_completed": 0,
                "total_epochs": epochs,
                "current_loss": None,
                "current_accuracy": None,
                "videos_processed": 0,
                "total_videos": 0
            })
        
        logger.info("Starting model retraining...")
        
        # Process training data
        logger.info("Processing training data...")
        X, y = process_training_data(data_dir, use_augmentation=True)
        
        with training_lock:
            training_status["progress"] = 20
            training_status["status"] = "splitting_data"
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        
        with training_lock:
            training_status["progress"] = 25
            training_status["status"] = "creating_model"
        
        # Create new model
        logger.info("Creating model architecture...")
        new_model = create_enhanced_model()
        
        # Calculate class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        logger.info(f"Class weights: {class_weight_dict}")
        
        with training_lock:
            training_status["progress"] = 30
            training_status["status"] = "training"
        
        # Setup callbacks
        callbacks = [
            TrainingProgressCallback(epochs),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1,
                cooldown=1
            ),
            ModelCheckpoint(
                filepath=TEMP_MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]
        
        # Train model
        logger.info("Starting training...")
        history = new_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        with training_lock:
            training_status["progress"] = 90
            training_status["status"] = "saving_model"
        
        # Replace the current model
        logger.info("Replacing current model...")
        if os.path.exists(TEMP_MODEL_PATH):
            # Backup old model
            if os.path.exists(MODEL_PATH):
                backup_path = MODEL_PATH.replace('.h5', '_backup.h5')
                shutil.copy2(MODEL_PATH, backup_path)
                logger.info(f"Backed up old model to {backup_path}")
            
            # Replace with new model
            shutil.move(TEMP_MODEL_PATH, MODEL_PATH)
            
            # Load new model
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("New model loaded successfully!")
        
        # Final evaluation
        val_loss, val_accuracy = new_model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
        
        with training_lock:
            training_status.update({
                "is_training": False,
                "progress": 100,
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "current_accuracy": float(val_accuracy),
                "current_loss": float(val_loss)
            })
        
        logger.info("Model retraining completed successfully!")
        
        # Clean up training data
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            logger.info("Cleaned up training data")
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        
        with training_lock:
            training_status.update({
                "is_training": False,
                "status": "failed",
                "error": error_msg,
                "completed_at": datetime.now().isoformat()
            })
        
        # Clean up on error
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        if os.path.exists(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    logger.info("Starting up Fight Detection API...")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    try:
        logger.info(f"Loading model from: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            logger.info("Available .h5 files in current directory:")
            for file in os.listdir(CURRENT_DIR):
                if file.endswith('.h5'):
                    logger.info(f"  - {file}")
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        model = load_model_with_custom_objects()
        
        if model is None:
            logger.warning("Creating fallback model...")
            model = create_fallback_model()
        else:
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {model.input_shape}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Creating fallback model...")
        model = create_fallback_model()
    
    yield
    
    logger.info("Shutting down Fight Detection API...")

app = FastAPI(
    title="Fight Detection API",
    description="API for detecting fights in video files using 3D CNN with retraining capability",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictionOutput(BaseModel):
    prediction: str = Field(..., description="Prediction label: 'fight' or 'noFight'")
    probability: float = Field(..., description="Probability of fight (0-1)")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    confidence_level: str = Field(..., description="Confidence level: 'High', 'Medium', or 'Low'")
    threshold_used: float = Field(..., description="Decision threshold used")
    frames_processed: int = Field(..., description="Number of frames processed")
    video_duration_estimate: Optional[float] = Field(None, description="Estimated video duration in seconds")
    model_status: str = Field(..., description="Status of the model used")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_input_shape: Optional[list] = None
    tensorflow_version: str
    opencv_version: str

class RetrainRequest(BaseModel):
    epochs: int = Field(default=20, ge=1, le=100, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, le=32, description="Training batch size")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Validation split ratio")

class RetrainResponse(BaseModel):
    message: str
    task_id: str
    status: str

class TrainingStatusResponse(BaseModel):
    is_training: bool
    progress: int = Field(..., ge=0, le=100, description="Training progress percentage")
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]
    task_id: Optional[str]
    epochs_completed: int
    total_epochs: int
    current_loss: Optional[float]
    current_accuracy: Optional[float]
    videos_processed: int
    total_videos: int

def extract_frames_improved(video_path, max_frames=FRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else None
    
    logger.info(f"Video info - Total frames: {total_frames}, FPS: {fps}, Duration: {duration}s")
    
    if total_frames == 0:
        cap.release()
        raise ValueError("Video has no frames")
    
    if total_frames < max_frames:
        step = 1
    else:
        step = total_frames // max_frames
    
    frame_indices = list(range(0, total_frames, step))[:max_frames]
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Could not read frame at index {frame_idx}")
            break
        
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)
        
        if len(frames) == max_frames:
            break
    
    cap.release()
    
    while len(frames) < max_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
    
    frames_array = np.array(frames[:max_frames])
    
    return frames_array, duration

async def predict_video_enhanced(video_path, threshold=0.5):
    global model
    
    if model is None:
        raise ValueError("Model is not loaded")
    
    try:
        loop = asyncio.get_event_loop()
        frames, duration = await loop.run_in_executor(
            None, extract_frames_improved, video_path, FRAMES_PER_VIDEO
        )
        
        if frames.shape[0] != FRAMES_PER_VIDEO:
            raise ValueError(f"Could not extract {FRAMES_PER_VIDEO} frames, got {frames.shape[0]}")
        
        input_array = np.expand_dims(frames.astype(np.float32) / 255.0, axis=0)
        
        prediction_prob = await loop.run_in_executor(
            None, lambda: model.predict(input_array, verbose=0)[0][0]
        )
        
        prediction_label = "fight" if prediction_prob > threshold else "noFight"
        
        confidence_score = max(prediction_prob, 1 - prediction_prob)
        if confidence_score > 0.8:
            confidence_level = "High"
        elif confidence_score > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        model_status = "original" if os.path.exists(MODEL_PATH) else "fallback"
        
        return {
            'prediction': prediction_label,
            'probability': float(prediction_prob),
            'confidence_score': float(confidence_score),
            'confidence_level': confidence_level,
            'threshold_used': threshold,
            'frames_processed': len(frames),
            'video_duration_estimate': duration,
            'model_status': model_status
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise ValueError(f"Prediction failed: {str(e)}")

# API Endpoints
@app.post("/predict", response_model=PredictionOutput, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_fight(
    file: UploadFile = File(..., description="Video file to analyze"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold for classification")
):
    """
    Predict whether a video contains fighting based on uploaded video file
    
    - **file**: Video file to analyze (supported formats: mp4, avi, mov, mkv, flv, wmv)
    - **threshold**: Decision threshold for classification (0.0 to 1.0, default: 0.5)
    """
    
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs and ensure the model file exists."
        )
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    file_extension = os.path.splitext(file.filename.lower())[1]
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type '{file_extension}'. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    MAX_FILE_SIZE = 100 * 1024 * 1024
    contents = await file.read()
    
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        logger.info(f"Processing video: {file.filename} (size: {len(contents)} bytes)")
        
        result = await predict_video_enhanced(temp_file_path, threshold)
        
        return PredictionOutput(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing video {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_file_path}: {e}")

@app.post("/retrain", response_model=RetrainResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def retrain_model(
    background_tasks: BackgroundTasks,
    training_data: UploadFile = File(..., description="ZIP file containing training data"),
    epochs: int = Form(20, description="Number of training epochs"),
    batch_size: int = Form(4, description="Training batch size"),
    validation_split: float = Form(0.2, description="Validation split ratio")
):
    """
    Retrain the model with new data
    
    Upload a ZIP file containing training data organized in folders:
    - normal/ (videos without fighting)
    - fight/ (videos with fighting)
    
    The retraining process runs in the background and can be monitored via /training-status
    """
    
    # Check if already training
    if training_status["is_training"]:
        raise HTTPException(
            status_code=400,
            detail="Model is already being retrained. Check /training-status for progress."
        )
    
    # Validate parameters
    if not (1 <= epochs <= 100):
        raise HTTPException(status_code=400, detail="Epochs must be between 1 and 100")
    
    if not (1 <= batch_size <= 32):
        raise HTTPException(status_code=400, detail="Batch size must be between 1 and 32")
    
    if not (0.1 <= validation_split <= 0.5):
        raise HTTPException(status_code=400, detail="Validation split must be between 0.1 and 0.5")
    
    # Check file type
    if not training_data.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Training data must be a ZIP file")
    
    # Check file size
    MAX_TRAINING_DATA_SIZE = 500 * 1024 * 1024  # 500MB
    contents = await training_data.read()
    
    if len(contents) > MAX_TRAINING_DATA_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Training data too large. Maximum size is {MAX_TRAINING_DATA_SIZE // (1024*1024)}MB"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create temporary directory for this training session
    session_data_dir = os.path.join(RETRAIN_DATA_DIR, task_id)
    os.makedirs(session_data_dir, exist_ok=True)
    
    try:
        # Save and extract ZIP file
        zip_path = os.path.join(session_data_dir, "training_data.zip")
        
        with open(zip_path, "wb") as f:
            f.write(contents)
        
        # Extract ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(session_data_dir)
        
        # Remove ZIP file
        os.remove(zip_path)
        
        # Validate folder structure
        normal_dir = os.path.join(session_data_dir, "normal")
        fight_dir = os.path.join(session_data_dir, "fight")
        
        if not (os.path.exists(normal_dir) and os.path.exists(fight_dir)):
            raise HTTPException(
                status_code=400,
                detail="ZIP file must contain 'normal' and 'fight' folders with video files"
            )
        
        # Count videos
        normal_videos = len([f for f in os.listdir(normal_dir) 
                           if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))])
        fight_videos = len([f for f in os.listdir(fight_dir) 
                          if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'))])
        
        if normal_videos == 0 or fight_videos == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough training data. Found {normal_videos} normal videos and {fight_videos} fight videos. Need at least 1 of each."
            )
        
        logger.info(f"Found {normal_videos} normal videos and {fight_videos} fight videos for training")
        
        # Start background training
        background_tasks.add_task(
            retrain_model_background,
            session_data_dir,
            epochs,
            batch_size,
            validation_split,
            task_id
        )
        
        return RetrainResponse(
            message=f"Model retraining started with {normal_videos + fight_videos} videos",
            task_id=task_id,
            status="started"
        )
        
    except HTTPException:
        # Clean up on validation error
        if os.path.exists(session_data_dir):
            shutil.rmtree(session_data_dir)
        raise
    except Exception as e:
        # Clean up on other errors
        if os.path.exists(session_data_dir):
            shutil.rmtree(session_data_dir)
        logger.error(f"Error processing training data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing training data: {str(e)}")

@app.get("/training-status", response_model=TrainingStatusResponse)
def get_training_status():
    """
    Get the current status of model retraining
    
    Returns detailed information about the training progress including:
    - Whether training is in progress
    - Current progress percentage
    - Number of epochs completed
    - Current loss and accuracy metrics
    - Any errors that occurred
    """
    
    with training_lock:
        return TrainingStatusResponse(**training_status)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint with detailed system information"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_input_shape=list(model.input_shape) if model else None,
        tensorflow_version=tf.__version__,
        opencv_version=cv2.__version__
    )

@app.get("/model-info")
def model_info():
    """Get detailed model information"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_loaded": True,
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "total_params": model.count_params(),
            "layers": len(model.layers),
            "model_type": str(type(model)),
            "frames_per_video": FRAMES_PER_VIDEO,
            "image_size": IMG_SIZE,
            "tensorflow_version": tf.__version__,
            "model_path": MODEL_PATH,
            "model_exists": os.path.exists(MODEL_PATH),
            "training_status": training_status["status"],
            "is_training": training_status["is_training"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.delete("/cancel-training")
def cancel_training():
    """
    Cancel ongoing training (if possible)
    
    Note: This will attempt to stop training, but may not be immediate
    due to the nature of TensorFlow training loops.
    """
    
    if not training_status["is_training"]:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    with training_lock:
        training_status.update({
            "is_training": False,
            "status": "cancelled",
            "error": "Training cancelled by user",
            "completed_at": datetime.now().isoformat()
        })
    
    # Clean up training data directories
    try:
        for item in os.listdir(RETRAIN_DATA_DIR):
            item_path = os.path.join(RETRAIN_DATA_DIR, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        if os.path.exists(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)
    except Exception as e:
        logger.warning(f"Error cleaning up training data: {e}")
    
    return {"message": "Training cancellation requested", "status": "cancelled"}

@app.get("/")
def read_root():
    """
    Root endpoint that provides basic API information
    """
    return {
        "message": "Fight Detection API with Retraining",
        "description": "Upload videos for fight detection and retrain the model with new data",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Upload video file for fight detection",
            "POST /retrain": "Upload training data and retrain the model",
            "GET /training-status": "Check training progress",
            "DELETE /cancel-training": "Cancel ongoing training",
            "GET /health": "Check API health status",
            "GET /model-info": "Get detailed model information",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "Alternative API documentation (ReDoc)"
        },
        "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"],
        "max_file_size": "100MB for prediction, 500MB for training data",
        "model_status": "loaded" if model else "not_loaded",
        "training_status": training_status["status"],
        "processing_info": {
            "frames_per_video": FRAMES_PER_VIDEO,
            "image_size": IMG_SIZE,
            "default_threshold": 0.5
        },
        "training_data_format": {
            "description": "Upload ZIP file with folder structure:",
            "structure": {
                "normal/": "Videos without fighting",
                "fight/": "Videos with fighting"
            },
            "supported_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
        },
        "access_urls": {
            "api_docs": "http://localhost:8000/docs",
            "health_check": "http://localhost:8000/health",
            "model_info": "http://localhost:8000/model-info",
            "training_status": "http://localhost:8000/training-status"
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Enhanced Fight Detection API...")
    print("📝 Access the API at: http://localhost:8000")
    print("📚 View API documentation at: http://localhost:8000/docs")
    print("🔍 Health check at: http://localhost:8000/health")
    print("🏋️ Training status at: http://localhost:8000/training-status")
    print("⚡ To stop the server: Press Ctrl+C")
    print("-" * 50)
    print("🎯 New Features:")
    print("  • POST /retrain - Upload training data and retrain model")
    print("  • GET /training-status - Monitor training progress")
    print("  • DELETE /cancel-training - Cancel ongoing training")
    print("📋 Training Data Format:")
    print("  • Upload ZIP file containing:")
    print("    - normal/ folder with non-fighting videos")
    print("    - fight/ folder with fighting videos")
    print("-" * 50)
    
    uvicorn.run(
        "app:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )