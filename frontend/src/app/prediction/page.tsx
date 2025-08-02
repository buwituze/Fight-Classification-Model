"use client";

import type React from "react";

import { useState, useRef, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Upload,
  Play,
  AlertCircle,
  Clock,
  FileVideo,
  X,
  RefreshCw,
} from "lucide-react";
import {
  api,
  type HealthResponse,
  type ModelInfoResponse,
  type PredictionResponse,
} from "@/lib/api";
import Layout from "@/components/layout";

export default function PredictionPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.5);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchInitialData = async () => {
    try {
      const [healthData, modelData] = await Promise.all([
        api.getHealth(),
        api.getModelInfo(),
      ]);
      setHealth(healthData);
      setModelInfo(modelData);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch model information"
      );
    } finally {
      setInitialLoading(false);
    }
  };

  useEffect(() => {
    fetchInitialData();
  }, []);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Clear previous results and errors
      setPrediction(null);
      setError(null);

      // Check file size
      if (
        modelInfo?.max_file_size_mb &&
        file.size > modelInfo.max_file_size_mb * 1024 * 1024
      ) {
        setError(
          `File size exceeds maximum limit of ${modelInfo.max_file_size_mb} MB`
        );
        setSelectedFile(null);
        return;
      }

      // Check file format - more robust checking
      const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;
      if (
        modelInfo?.supported_formats &&
        !modelInfo.supported_formats.includes(fileExtension)
      ) {
        setError(
          `Unsupported file format "${fileExtension}". Supported formats: ${modelInfo.supported_formats.join(
            ", "
          )}`
        );
        setSelectedFile(null);
        return;
      }

      // Additional MIME type validation
      const supportedMimeTypes = [
        "video/mp4",
        "video/avi",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
        "video/x-flv",
        "video/x-ms-wmv",
        "video/webm",
      ];
      if (
        !supportedMimeTypes.some((type) =>
          file.type.includes(type.split("/")[1])
        )
      ) {
        console.warn(`File MIME type "${file.type}" might not be supported`);
      }

      setSelectedFile(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const result = await api.predict(selectedFile, threshold);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const handleClearFile = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleRetry = () => {
    setError(null);
    fetchInitialData();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600";
    if (confidence >= 0.6) return "text-yellow-600";
    return "text-red-600";
  };

  const getConfidenceBadgeVariant = (level: string) => {
    switch (level.toLowerCase()) {
      case "high":
        return "default";
      case "medium":
        return "secondary";
      case "low":
        return "destructive";
      default:
        return "outline";
    }
  };

  const getPredictionBadgeVariant = (prediction: string) => {
    return prediction === "fight" ? "destructive" : "default";
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  // Loading state for initial data fetch
  if (initialLoading) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Video Prediction
            </h1>
            <p className="text-gray-600">
              Upload a video to analyze and get AI predictions
            </p>
          </div>
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
            <span className="ml-2 text-gray-600">
              Loading model information...
            </span>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Video Prediction</h1>
          <p className="text-gray-600">
            Upload a video to analyze and get AI predictions
          </p>
        </div>

        {/* Model Status Check */}
        {health && !health.model_loaded && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>
                Model is not currently loaded. Please check the dashboard or
                contact support.
              </span>
              <Button variant="outline" size="sm" onClick={handleRetry}>
                <RefreshCw className="h-4 w-4 mr-1" />
                Retry
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {/* Global Error */}
        {error && !selectedFile && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>{error}</span>
              <Button variant="outline" size="sm" onClick={handleRetry}>
                <RefreshCw className="h-4 w-4 mr-1" />
                Retry
              </Button>
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
                  <span className="block mt-1 text-xs text-gray-500">
                    Max size: {modelInfo.max_file_size_mb} MB | Formats:{" "}
                    {modelInfo.supported_formats?.join(", ") || "Loading..."}
                  </span>
                )}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-center w-full">
                <label
                  htmlFor="video-upload"
                  className={`flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer transition-colors ${
                    selectedFile
                      ? "border-blue-300 bg-blue-50 hover:bg-blue-100"
                      : "border-gray-300 bg-gray-50 hover:bg-gray-100"
                  }`}
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    {selectedFile ? (
                      <>
                        <FileVideo className="w-10 h-10 mb-3 text-blue-500" />
                        <p className="mb-2 text-sm text-gray-700">
                          <span className="font-semibold">
                            {selectedFile.name}
                          </span>
                        </p>
                        <p className="text-xs text-gray-500 mb-2">
                          {formatFileSize(selectedFile.size)}
                        </p>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={(e) => {
                            e.preventDefault();
                            handleClearFile();
                          }}
                          className="mt-2"
                        >
                          <X className="w-3 h-3 mr-1" />
                          Remove
                        </Button>
                      </>
                    ) : (
                      <>
                        <Upload className="w-10 h-10 mb-3 text-gray-400" />
                        <p className="mb-2 text-sm text-gray-500">
                          <span className="font-semibold">Click to upload</span>{" "}
                          or drag and drop
                        </p>
                        <p className="text-xs text-gray-500">
                          Video files only
                        </p>
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
                    disabled={loading}
                  />
                </label>
              </div>

              <div className="space-y-2">
                <Label htmlFor="threshold">
                  Confidence Threshold: {threshold.toFixed(1)}
                  <span className="text-xs text-gray-500 ml-2">
                    (Higher = more strict)
                  </span>
                </Label>
                <Input
                  id="threshold"
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={threshold}
                  onChange={(e) =>
                    setThreshold(Number.parseFloat(e.target.value))
                  }
                  className="w-full"
                  disabled={loading}
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>0.0 (Permissive)</span>
                  <span>0.5 (Balanced)</span>
                  <span>1.0 (Strict)</span>
                </div>
              </div>

              <Button
                onClick={handlePredict}
                disabled={!selectedFile || loading || !health?.model_loaded}
                className="w-full"
                size="lg"
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

              {/* File-specific Error */}
              {error && selectedFile && (
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
              <CardDescription>
                AI analysis results and confidence scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading && (
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4" />
                    <p className="text-sm text-gray-600">
                      Processing video frames...
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      This may take a few moments
                    </p>
                  </div>
                  <Progress value={undefined} className="w-full" />
                </div>
              )}

              {prediction && !loading && (
                <div className="space-y-6">
                  {/* Main Prediction */}
                  <div className="text-center p-6 bg-gray-50 rounded-lg">
                    <h3 className="text-lg font-semibold mb-2">Prediction</h3>
                    <div className="mb-3">
                      <Badge
                        variant={getPredictionBadgeVariant(
                          prediction.prediction
                        )}
                        className="text-2xl font-bold py-2 px-4"
                      >
                        {prediction.prediction === "fight"
                          ? "🥊 Fight Detected"
                          : "✅ No Fight"}
                      </Badge>
                    </div>
                    <Badge
                      variant={getConfidenceBadgeVariant(
                        prediction.confidence_level
                      )}
                    >
                      {prediction.confidence_level} Confidence
                    </Badge>
                  </div>

                  {/* Confidence Score */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">
                        Confidence Score
                      </span>
                      <span
                        className={`text-sm font-bold ${getConfidenceColor(
                          prediction.confidence_score
                        )}`}
                      >
                        {(prediction.confidence_score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={prediction.confidence_score * 100}
                      className="w-full"
                    />
                  </div>

                  {/* Probability Details */}
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">
                        Fight Probability
                      </span>
                      <span className="text-sm font-bold">
                        {(prediction.probability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={prediction.probability * 100}
                      className="w-full"
                    />
                    <p className="text-xs text-gray-500">
                      Threshold used:{" "}
                      {(prediction.threshold_used * 100).toFixed(0)}%
                    </p>
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
                      <span className="font-medium">
                        {prediction.frames_processed}
                      </span>
                    </div>
                  </div>

                  {/* Model Status */}
                  <div className="text-center text-xs text-gray-500">
                    Model Status:{" "}
                    <span className="font-medium">
                      {prediction.model_status}
                    </span>
                  </div>

                  {/* Confidence Level Explanation */}
                  <div className="p-4 bg-blue-50 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">
                      Understanding Results
                    </h4>
                    <div className="text-sm text-blue-800 space-y-1">
                      <div>
                        <strong>High Confidence (≥80%):</strong> Very reliable
                        prediction
                      </div>
                      <div>
                        <strong>Medium Confidence (60-79%):</strong> Moderately
                        reliable
                      </div>
                      <div>
                        <strong>Low Confidence (&lt;60%):</strong> Less
                        reliable, consider manual review
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      onClick={handleClearFile}
                      className="flex-1"
                    >
                      Upload New Video
                    </Button>
                    <Button
                      variant="outline"
                      onClick={handlePredict}
                      disabled={loading}
                      className="flex-1"
                    >
                      <RefreshCw className="w-4 h-4 mr-1" />
                      Re-analyze
                    </Button>
                  </div>
                </div>
              )}

              {!prediction && !loading && (
                <div className="text-center py-12 text-gray-500">
                  <FileVideo className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="text-base mb-2">
                    Upload a video to see prediction results
                  </p>
                  <p className="text-sm">
                    {health?.model_loaded
                      ? "Model is ready for analysis"
                      : "Waiting for model to load..."}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </Layout>
  );
}
