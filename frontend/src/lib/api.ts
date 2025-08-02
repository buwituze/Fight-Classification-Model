const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL!

console.log('API_BASE_URL:', API_BASE_URL)


export interface HealthResponse {
  status: string
  model_loaded: boolean
  model_input_shape: number[] | null
  tensorflow_version: string
  opencv_version: string
  sklearn_available: boolean
  training_available: boolean
  uptime: string
}

export interface TrainingStatusResponse {
  is_training: boolean
  progress: number
  status: string
  started_at: string | null
  completed_at: string | null
  error: string | null
  task_id: string | null
  epochs_completed: number
  total_epochs: number
  current_loss: number | null
  current_accuracy: number | null
  videos_processed: number
  total_videos: number
  estimated_time_remaining: string | null
}

export interface ModelInfoResponse {
  model_loaded: boolean
  input_shape: number[]
  output_shape: number[]
  total_params: number
  trainable_params: number
  layers: number
  model_type: string
  frames_per_video: number
  image_size: number
  tensorflow_version: string
  model_path: string
  model_exists: boolean
  training_status: string
  is_training: boolean
  sklearn_available: boolean
  supported_formats: string[]
  max_file_size_mb: number
  max_training_data_size_mb: number
}

export interface PredictionResponse {
  prediction: string
  probability: number
  confidence_score: number
  confidence_level: string
  threshold_used: number
  frames_processed: number
  video_duration_estimate: number | null
  model_status: string
}

export interface DatasetStatsResponse {
  model_performance_history: Array<{
    version: string
    date: string
    accuracy: number
    precision: number
    recall: number
    f1_score: number
  }>
  data_distribution: {
    fight_videos: number
    normal_videos: number
    total_videos: number
    fight_ratio: number
    normal_ratio: number
  }
  training_summary: {
    total_models_trained: number
    latest_accuracy: number
    improvement: number
  }
}

export interface ConfidenceAnalysisResponse {
  confidence_distribution: {
    high_confidence: number
    medium_confidence: number
    low_confidence: number
    description: string
    ranges: string[]
    counts: number[]
  }
  threshold_analysis: {
    current_threshold: number
    optimal_threshold: number
    description: string
    thresholds: number[]
    accuracy_scores: number[]
  }
  prediction_patterns: {
    average_confidence_score: number
    fight_detection_confidence: number
    normal_detection_confidence: number
    description: string
    high_confidence: number
    medium_confidence: number
    low_confidence: number
  }
  model_characteristics: {
    frames_processed_per_video: number
    image_size: number
    model_status: string
    description: string
  }
}

export type AnalyticsResponse = DatasetStatsResponse

const handleApiCall = async (endpoint: string, options: RequestInit = {}): Promise<any> => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        ...options.headers,
      },
    })

    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}: ${response.statusText}`
      try {
        const errorData = await response.json()
        errorMessage = errorData.detail || errorData.error || errorMessage
      } catch {
        // If we can't parse the error response, use the default message
      }
      throw new Error(errorMessage)
    }

    return await response.json()
  } catch (error) {
    console.error(`API call to ${endpoint} failed:`, error)
    throw error
  }
}

export const api = {
  // Dashboard endpoints
  getHealth: (): Promise<HealthResponse> => handleApiCall("/health"),
  getTrainingStatus: (): Promise<TrainingStatusResponse> => handleApiCall("/training-status"),
  getModelInfo: (): Promise<ModelInfoResponse> => handleApiCall("/model-info"),

  // Prediction endpoints
  predict: (file: File, threshold = 0.5): Promise<PredictionResponse> => {
    const formData = new FormData()
    formData.append("file", file)
    formData.append("threshold", threshold.toString())

    return handleApiCall("/predict", {
      method: "POST",
      body: formData,
    })
  },

  // Training endpoints
  retrain: (trainingData: File, epochs = 20, batchSize = 4, validationSplit = 0.2) => {
    const formData = new FormData()
    formData.append("training_data", trainingData)
    formData.append("epochs", epochs.toString())
    formData.append("batch_size", batchSize.toString())
    formData.append("validation_split", validationSplit.toString())

    return handleApiCall("/retrain", {
      method: "POST",
      body: formData,
    })
  },

  cancelTraining: () => handleApiCall("/cancel-training", { method: "DELETE" }),

  // Analytics endpoints
  getDatasetStats: (): Promise<DatasetStatsResponse> => handleApiCall("/analytics/dataset-stats"),
  getConfidenceAnalysis: (): Promise<ConfidenceAnalysisResponse> => handleApiCall("/analytics/confidence-analysis"),
}
