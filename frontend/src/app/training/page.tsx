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
import { Separator } from "@/components/ui/separator"
import { Upload, Play, Square, AlertCircle, FileArchive, Settings, TrendingUp } from "lucide-react"
import { api, type TrainingStatusResponse, type ModelInfoResponse } from "@/lib/api"
import Layout from "@/components/layout"

export default function TrainingPage() {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatusResponse | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [epochs, setEpochs] = useState(20)
  const [batchSize, setBatchSize] = useState(4)
  const [validationSplit, setValidationSplit] = useState(0.2)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const fetchTrainingStatus = async () => {
    try {
      const status = await api.getTrainingStatus()
      setTrainingStatus(status)
    } catch (err) {
      console.error("Failed to fetch training status:", err)
    }
  }

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [statusData, modelData] = await Promise.all([api.getTrainingStatus(), api.getModelInfo()])
        setTrainingStatus(statusData)
        setModelInfo(modelData)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to fetch training information")
      }
    }

    fetchInitialData()

    // Poll for training status every 2 seconds when training is active
    const interval = setInterval(() => {
      if (trainingStatus?.is_training) {
        fetchTrainingStatus()
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [trainingStatus?.is_training])

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      // Check if it's a ZIP file
      if (!file.name.toLowerCase().endsWith(".zip")) {
        setError("Please select a ZIP file containing training data")
        return
      }

      setSelectedFile(file)
      setError(null)
    }
  }

  const handleStartTraining = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    try {
      await api.retrain(selectedFile, epochs, batchSize, validationSplit)
      // Immediately fetch updated status
      await fetchTrainingStatus()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training")
    } finally {
      setLoading(false)
    }
  }

  const handleCancelTraining = async () => {
    try {
      await api.cancelTraining()
      await fetchTrainingStatus()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to cancel training")
    }
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case "training":
        return "default"
      case "completed":
        return "default"
      case "failed":
        return "destructive"
      default:
        return "secondary"
    }
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Training</h1>
          <p className="text-gray-600">Upload training data and retrain your AI model</p>
        </div>

        {/* Current Training Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <TrendingUp className="h-5 w-5" />
              <span>Training Status</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold mb-1">
                  <Badge variant={getStatusBadgeVariant(trainingStatus?.status || "idle")}>
                    {trainingStatus?.status || "idle"}
                  </Badge>
                </div>
                <p className="text-sm text-gray-600">Current Status</p>
              </div>

              {trainingStatus?.is_training && (
                <>
                  <div className="text-center">
                    <div className="text-2xl font-bold mb-1">{trainingStatus.progress}%</div>
                    <p className="text-sm text-gray-600">Progress</p>
                  </div>

                  <div className="text-center">
                    <div className="text-2xl font-bold mb-1">
                      {trainingStatus.current_epoch || 0}/{trainingStatus.total_epochs || 0}
                    </div>
                    <p className="text-sm text-gray-600">Epochs</p>
                  </div>
                </>
              )}
            </div>

            {trainingStatus?.is_training && (
              <div className="mt-4 space-y-2">
                <Progress value={trainingStatus.progress} className="w-full" />
                {trainingStatus.loss !== undefined && trainingStatus.accuracy !== undefined && (
                  <div className="flex justify-between text-sm text-gray-600">
                    <span>Loss: {trainingStatus.loss.toFixed(4)}</span>
                    <span>Accuracy: {(trainingStatus.accuracy * 100).toFixed(2)}%</span>
                  </div>
                )}
              </div>
            )}

            {trainingStatus?.is_training && (
              <div className="mt-4">
                <Button variant="destructive" onClick={handleCancelTraining} className="w-full">
                  <Square className="w-4 h-4 mr-2" />
                  Cancel Training
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Training Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <CardDescription>Upload training data and set parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* File Upload */}
              <div className="space-y-2">
                <Label>Training Data (ZIP file)</Label>
                <div className="flex items-center justify-center w-full">
                  <label
                    htmlFor="training-upload"
                    className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 hover:bg-gray-100"
                  >
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      {selectedFile ? (
                        <>
                          <FileArchive className="w-8 h-8 mb-2 text-gray-400" />
                          <p className="text-sm text-gray-500">
                            <span className="font-semibold">{selectedFile.name}</span>
                          </p>
                          <p className="text-xs text-gray-500">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                        </>
                      ) : (
                        <>
                          <Upload className="w-8 h-8 mb-2 text-gray-400" />
                          <p className="text-sm text-gray-500">
                            <span className="font-semibold">Click to upload</span> ZIP file
                          </p>
                        </>
                      )}
                    </div>
                    <Input
                      id="training-upload"
                      type="file"
                      className="hidden"
                      accept=".zip"
                      onChange={handleFileSelect}
                      ref={fileInputRef}
                    />
                  </label>
                </div>
              </div>

              <Separator />

              {/* Training Parameters */}
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Settings className="h-4 w-4 text-gray-400" />
                  <Label className="text-base font-medium">Training Parameters</Label>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="epochs">Epochs</Label>
                    <Input
                      id="epochs"
                      type="number"
                      min="1"
                      max="100"
                      value={epochs}
                      onChange={(e) => setEpochs(Number.parseInt(e.target.value))}
                      disabled={trainingStatus?.is_training}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="batch-size">Batch Size</Label>
                    <Input
                      id="batch-size"
                      type="number"
                      min="1"
                      max="32"
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number.parseInt(e.target.value))}
                      disabled={trainingStatus?.is_training}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="validation-split">Validation Split: {(validationSplit * 100).toFixed(0)}%</Label>
                  <Input
                    id="validation-split"
                    type="range"
                    min="0.1"
                    max="0.5"
                    step="0.05"
                    value={validationSplit}
                    onChange={(e) => setValidationSplit(Number.parseFloat(e.target.value))}
                    disabled={trainingStatus?.is_training}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>10%</span>
                    <span>50%</span>
                  </div>
                </div>
              </div>

              <Button
                onClick={handleStartTraining}
                disabled={!selectedFile || loading || trainingStatus?.is_training}
                className="w-full"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Start Training
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

          {/* Training Information */}
          <Card>
            <CardHeader>
              <CardTitle>Training Information</CardTitle>
              <CardDescription>Current model details and requirements</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Current Model Info */}
              <div className="space-y-4">
                <h4 className="font-medium">Current Model</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Parameters:</span>
                    <span className="font-medium">{modelInfo?.total_params?.toLocaleString() || "N/A"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Input Shape:</span>
                    <span className="font-medium">{modelInfo?.input_shape?.join(" × ") || "N/A"}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Supported Formats:</span>
                    <span className="font-medium">{modelInfo?.supported_formats?.join(", ") || "N/A"}</span>
                  </div>
                </div>
              </div>

              <Separator />

              {/* Training Requirements */}
              <div className="space-y-4">
                <h4 className="font-medium">Training Data Requirements</h4>
                <div className="p-4 bg-blue-50 rounded-lg">
                  <ul className="text-sm text-blue-800 space-y-2">
                    <li>• ZIP file containing training videos</li>
                    <li>• Organize videos in folders by class (fight/normal)</li>
                    <li>• Minimum 50 videos per class recommended</li>
                    <li>• Video formats: {modelInfo?.supported_formats?.join(", ") || "MP4, AVI, MOV"}</li>
                    <li>• Maximum file size per video: {modelInfo?.max_file_size_mb || 100}MB</li>
                  </ul>
                </div>
              </div>

              <Separator />

              {/* Training Tips */}
              <div className="space-y-4">
                <h4 className="font-medium">Training Tips</h4>
                <div className="p-4 bg-green-50 rounded-lg">
                  <ul className="text-sm text-green-800 space-y-2">
                    <li>• Higher epochs = better accuracy but longer training</li>
                    <li>• Smaller batch size = more stable training</li>
                    <li>• 20% validation split is recommended</li>
                    <li>• Training typically takes 30-60 minutes</li>
                    <li>• Monitor loss and accuracy during training</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  )
}
