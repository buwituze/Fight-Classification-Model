"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  AlertCircle,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  Zap,
} from "lucide-react";
import {
  api,
  type HealthResponse,
  type TrainingStatusResponse,
  type ModelInfoResponse,
} from "@/lib/api";
import Layout from "@/components/layout";

export default function Dashboard() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [trainingStatus, setTrainingStatus] =
    useState<TrainingStatusResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      const [healthData, trainingData, modelData] = await Promise.all([
        api.getHealth(),
        api.getTrainingStatus(),
        api.getModelInfo(),
      ]);

      setHealth(healthData);
      setTrainingStatus(trainingData);
      setModelInfo(modelData);
      setError(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch dashboard data"
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();

    // Poll for updates every 30 seconds
    const interval = setInterval(fetchDashboardData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fixed uptime formatting - the API returns a string like "1:23:45", not seconds
  const formatUptime = (uptime: string) => {
    if (!uptime || uptime === "unknown") return "Unknown";
    return uptime; // Already formatted as "Xh Ym" or "X:Y:Z"
  };

  const formatFileSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  // Helper function to get training status color
  const getTrainingStatusVariant = (status: string) => {
    switch (status) {
      case "training":
      case "preparing_data":
      case "splitting_data":
      case "creating_model":
        return "default";
      case "completed":
        return "default";
      case "failed":
      case "cancelled":
        return "destructive";
      case "idle":
      default:
        return "secondary";
    }
  };

  if (loading) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
            <p className="text-gray-600">
              Monitor your AI model status and performance
            </p>
          </div>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {[...Array(6)].map((_, i) => (
              <Card key={i}>
                <CardHeader>
                  <Skeleton className="h-4 w-24" />
                </CardHeader>
                <CardContent>
                  <Skeleton className="h-8 w-16 mb-2" />
                  <Skeleton className="h-4 w-32" />
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
            <p className="text-gray-600">
              Monitor your AI model status and performance
            </p>
          </div>
          <Card>
            <CardContent className="flex items-center space-x-2 pt-6">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <span className="text-red-700">{error}</span>
            </CardContent>
          </Card>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600">
            Monitor your AI model status and performance
          </p>
        </div>

        {/* Status Cards */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {/* Model Status */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Model Status
              </CardTitle>
              {health?.model_loaded ? (
                <CheckCircle className="h-4 w-4 text-green-600" />
              ) : (
                <AlertCircle className="h-4 w-4 text-red-600" />
              )}
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                <Badge
                  variant={health?.model_loaded ? "default" : "destructive"}
                >
                  {health?.status || "Unknown"}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                {health?.model_loaded
                  ? "Model is ready for predictions"
                  : "Model not available"}
              </p>
            </CardContent>
          </Card>

          {/* System Uptime - FIXED */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                System Uptime
              </CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {health?.uptime ? formatUptime(health.uptime) : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                Continuous operation time
              </p>
            </CardContent>
          </Card>

          {/* Training Status - IMPROVED */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Training Status
              </CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                <Badge
                  variant={getTrainingStatusVariant(
                    trainingStatus?.status || "idle"
                  )}
                >
                  {trainingStatus?.status || "idle"}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">
                {trainingStatus?.is_training
                  ? `Progress: ${trainingStatus.progress}% (${trainingStatus.epochs_completed}/${trainingStatus.total_epochs} epochs)`
                  : health?.training_available
                  ? "Ready for training"
                  : "Training not available"}
              </p>
            </CardContent>
          </Card>

          {/* Model Parameters */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Model Parameters
              </CardTitle>
              <Cpu className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {modelInfo?.total_params?.toLocaleString() || "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                Total trainable parameters ({modelInfo?.layers || 0} layers)
              </p>
            </CardContent>
          </Card>

          {/* File Size Limit */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Max File Size
              </CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {modelInfo?.max_file_size_mb
                  ? `${modelInfo.max_file_size_mb} MB`
                  : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">
                Maximum upload size
              </p>
            </CardContent>
          </Card>

          {/* Supported Formats */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Supported Formats
              </CardTitle>
              <Database className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {modelInfo?.supported_formats?.length || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                {modelInfo?.supported_formats?.slice(0, 3).join(", ") ||
                  "No formats available"}
                {modelInfo?.supported_formats &&
                  modelInfo.supported_formats.length > 3 &&
                  "..."}
              </p>
            </CardContent>
          </Card>
        </div>

        {/* System Information */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>System Information</CardTitle>
              <CardDescription>
                Current system versions and capabilities
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between">
                <span className="text-sm font-medium">TensorFlow Version:</span>
                <span className="text-sm text-muted-foreground">
                  {health?.tensorflow_version || "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">OpenCV Version:</span>
                <span className="text-sm text-muted-foreground">
                  {health?.opencv_version || "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">
                  scikit-learn Available:
                </span>
                <Badge
                  variant={health?.sklearn_available ? "default" : "secondary"}
                >
                  {health?.sklearn_available ? "Yes" : "No"}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Training Available:</span>
                <Badge
                  variant={health?.training_available ? "default" : "secondary"}
                >
                  {health?.training_available ? "Yes" : "No"}
                </Badge>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model Configuration</CardTitle>
              <CardDescription>Current model specifications</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Input Shape:</span>
                <span className="text-sm text-muted-foreground">
                  {health?.model_input_shape?.join(" × ") ||
                    modelInfo?.input_shape?.join(" × ") ||
                    "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Total Parameters:</span>
                <span className="text-sm text-muted-foreground">
                  {modelInfo?.total_params?.toLocaleString() || "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">
                  Trainable Parameters:
                </span>
                <span className="text-sm text-muted-foreground">
                  {modelInfo?.trainable_params?.toLocaleString() || "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Model Type:</span>
                <span className="text-sm text-muted-foreground">
                  {modelInfo?.model_type || "N/A"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm font-medium">Status:</span>
                <Badge
                  variant={health?.model_loaded ? "default" : "destructive"}
                >
                  {health?.model_loaded ? "Loaded" : "Not Loaded"}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Training Progress (if training is active) */}
        {trainingStatus?.is_training && (
          <Card>
            <CardHeader>
              <CardTitle>Training Progress</CardTitle>
              <CardDescription>
                Current training session details
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between">
                <span className="text-sm font-medium">Progress:</span>
                <span className="text-sm text-muted-foreground">
                  {trainingStatus.progress}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${trainingStatus.progress}%` }}
                ></div>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="flex justify-between">
                  <span className="font-medium">Epochs:</span>
                  <span className="text-muted-foreground">
                    {trainingStatus.epochs_completed}/
                    {trainingStatus.total_epochs}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium">Videos:</span>
                  <span className="text-muted-foreground">
                    {trainingStatus.videos_processed}/
                    {trainingStatus.total_videos}
                  </span>
                </div>
                {trainingStatus.current_accuracy && (
                  <div className="flex justify-between">
                    <span className="font-medium">Accuracy:</span>
                    <span className="text-muted-foreground">
                      {(trainingStatus.current_accuracy * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
                {trainingStatus.current_loss && (
                  <div className="flex justify-between">
                    <span className="font-medium">Loss:</span>
                    <span className="text-muted-foreground">
                      {trainingStatus.current_loss.toFixed(4)}
                    </span>
                  </div>
                )}
              </div>
              {trainingStatus.estimated_time_remaining && (
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Time Remaining:</span>
                  <span className="text-sm text-muted-foreground">
                    {trainingStatus.estimated_time_remaining}
                  </span>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}
