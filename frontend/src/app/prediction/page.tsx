"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, Play, AlertCircle, Clock, FileVideo } from "lucide-react"
import { api, type HealthResponse, type ModelInfoResponse, type PredictionResponse } from "@/lib/api"
import Layout from "@/components/layout"

export default function PredictionPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [threshold, setThreshold] = useState(0.5)
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [healthData, modelData] = await Promise.all([api.getHealth(), api.getModelInfo()])
        setHealth(healthData)
        setModelInfo(modelData)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch model information")
      }
    }

    fetchInitialData()
  }, [])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Check file size
      if (modelInfo?.max_file_size_mb && file.size > modelInfo.max_file_size_mb * 1024 * 1024) {
        setError(`File size exceeds maximum limit of ${modelInfo.max_file_size_mb} MB`)
        return
      }

      // Check file format
      const fileExtension = file.name.split(".").pop()?.toLowerCase()
      if (modelInfo?.supported_formats && !modelInfo.supported_formats.includes(`.${fileExtension}`)) {
        setError(`Unsupported file format. Supported formats: ${modelInfo.supported_formats.join(", ")}`)
        return
      }

      setSelectedFile(file)
      setPrediction(null)
      setError(null)
    }
  }

  const handlePredict = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    try {
      const result = await api.predict(selectedFile, threshold)
      setPrediction(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed")
    } finally {
      setLoading(false)
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600"
    if (confidence >= 0.6) return "text-yellow-600"
    return "text-red-600"
  }

  const getConfidenceBadgeVariant = (level: string) => {
    switch (level.toLowerCase()) {
      case "high":
        return "default"
      case "medium":
        return "secondary"
      case "low":
        return "destructive"
      default:
        return "outline"
    }
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Video Prediction</h1>
          <p className="text-gray-600">Upload a video to analyze and get AI predictions</p>
        </div>

        {/* Model Status Check */}
        {health && !health.model_loaded && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Model is not currently loaded. Please check the dashboard or contact support.
            </AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle>Upload Video</CardTitle>
              <CardDescription>
                Select a video file for analysis
                {modelInfo && (
                  <span className="block mt-1 text-xs">
                    Max size: {modelInfo.max_file_size_mb} MB | Formats: {modelInfo.supported_formats.join(", ")}
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-center w-full">
                <label
                  htmlFor="video-upload"
                  className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    {selectedFile ? (
                      <>
                        <FileVideo className="w-10 h-10 mb-3 text-gray-400" />
                        <p className="mb-2 text-sm text-gray-500">
                          <span className="font-semibold">{selectedFile.name}</span>
                        </p>
                        <p className="text-xs text-gray-500">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                      </>
                    ) : (
                      <>
                        <Upload className="w-10 h-10 mb-3 text-gray-400" />
                        <p className="mb-2 text-sm text-gray-500">
                          <span className="font-semibold">Click to upload</span> or drag and drop
                        </p>
                        <p className="text-xs text-gray-500">Video files only</p>
                      </>
                    )}
                  </div>
                  <Input
                    id="video-upload"
                    type="file"
                    className="hidden"
                    accept="video/*"
                    onChange={handleFileSelect}
                    ref={fileInputRef}
                  />
                </label>
              </div>

              <div className="space-y-2">
                <Label htmlFor="threshold">Confidence Threshold: {threshold}</Label>
                <Input
                  id="threshold"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={threshold}
                  onChange={(e) => setThreshold(Number.parseFloat(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0.0 (Low)</span>
                  <span>1.0 (High)</span>
                </div>
              </div>

              <Button
                onClick={handlePredict}
                disabled={!selectedFile || loading || !health?.model_loaded}
                className="w-full"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Analyze Video
                  </>
                )}
              </Button>

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card>
            <CardHeader>
              <CardTitle>Prediction Results</CardTitle>
              <CardDescription>AI analysis results and confidence scores</CardDescription>
            </CardHeader>
            <CardContent>
              {loading && (
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4" />
                    <p className="text-sm text-gray-600">Processing video...</p>
                  </div>
                  <Progress value={undefined} className="w-full" />
                </div>
              )}

              {prediction && !loading && (
                <div className="space-y-6">
                  {/* Main Prediction */}
                  <div className="text-center p-6 bg-gray-50 rounded-lg">
                    <h3 className="text-lg font-semibold mb-2">Prediction</h3>
                    <div className="text-3xl font-bold mb-2">{prediction.prediction}</div>
                    <Badge variant={getConfidenceBadgeVariant(prediction.confidence_level)}>
                      {prediction.confidence_level} Confidence
                    </Badge>
                  </div>

                  {/* Confidence Score */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Confidence Score</span>
                      <span className={`text-sm font-bold ${getConfidenceColor(prediction.confidence_score)}`}>
                        {(prediction.confidence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress value={prediction.confidence_score * 100} className="w-full" />
                  </div>

                  {/* Processing Details */}
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-gray-400" />
                      <span className="text-gray-600">Duration:</span>
                      <span className="font-medium">
                        {prediction.video_duration_estimate
                          ? `${prediction.video_duration_estimate.toFixed(1)}s`
                          : "N/A"}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <FileVideo className="h-4 w-4 text-gray-400" />
                      <span className="text-gray-600">Frames:</span>
                      <span className="font-medium">{prediction.frames_processed}</span>
                    </div>
                  </div>

                  {/* Confidence Level Explanation */}
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">Confidence Level Guide</h4>
                    <div className="text-sm text-blue-800 space-y-1">
                      <div>
                        <strong>High (≥80%):</strong> Very reliable prediction
                      </div>
                      <div>
                        <strong>Medium (60-79%):</strong> Moderately reliable prediction
                      </div>
                      <div>
                        <strong>Low (&lt;60%):</strong> Less reliable, consider retraining
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {!prediction && !loading && (
                <div className="text-center py-12 text-gray-500">
                  <FileVideo className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Upload a video to see prediction results</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  )
}
