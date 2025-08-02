"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from "recharts"
import { Clock, Target, AlertCircle, Activity, Zap } from "lucide-react"
import { api, type AnalyticsResponse, type ConfidenceAnalysisResponse, type HealthResponse } from "@/lib/api"
import Layout from "@/components/layout"

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"]

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null)
  const [confidenceAnalysis, setConfidenceAnalysis] = useState<ConfidenceAnalysisResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAnalyticsData = async () => {
    try {
      const [analyticsData, confidenceData, healthData] = await Promise.all([
        api.getDatasetStats(),
        api.getConfidenceAnalysis(),
        api.getHealth(),
      ])

      setAnalytics(analyticsData)
      setConfidenceAnalysis(confidenceData)
      setHealth(healthData)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch analytics data")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAnalyticsData()

    // Refresh analytics every 5 minutes
    const interval = setInterval(fetchAnalyticsData, 300000)
    return () => clearInterval(interval)
  }, [])

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400)
    const hours = Math.floor((seconds % 86400) / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)

    if (days > 0) return `${days}d ${hours}h`
    if (hours > 0) return `${hours}h ${minutes}m`
    return `${minutes}m`
  }

  // Prepare chart data
  const performanceData = analytics?.model_performance_history
    ? analytics.model_performance_history.map((item) => ({
        date: new Date(item.date).toLocaleDateString(),
        accuracy: (item.accuracy * 100).toFixed(1),
        precision: (item.precision * 100).toFixed(1),
        recall: (item.recall * 100).toFixed(1),
        f1_score: (item.f1_score * 100).toFixed(1),
      }))
    : []

  const dataDistributionData = analytics?.data_distribution
    ? [
        { name: "Fight Videos", value: analytics.data_distribution.fight_videos, color: "#FF8042" },
        { name: "Normal Videos", value: analytics.data_distribution.normal_videos, color: "#00C49F" },
      ]
    : []

  const confidenceDistributionData = confidenceAnalysis?.confidence_distribution
    ? confidenceAnalysis.confidence_distribution.ranges.map((range, index) => ({
        range,
        count: confidenceAnalysis.confidence_distribution.counts[index],
      }))
    : []

  const thresholdAnalysisData = confidenceAnalysis?.threshold_analysis
    ? confidenceAnalysis.threshold_analysis.thresholds.map((threshold, index) => ({
        threshold: threshold.toFixed(1),
        accuracy: (confidenceAnalysis.threshold_analysis.accuracy_scores[index] * 100).toFixed(1),
      }))
    : []

  if (loading) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
            <p className="text-gray-600">Model performance insights and data analysis</p>
          </div>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => (
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
    )
  }

  if (error) {
    return (
      <Layout>
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
            <p className="text-gray-600">Model performance insights and data analysis</p>
          </div>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      </Layout>
    )
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Analytics</h1>
          <p className="text-gray-600">Model performance insights and data analysis</p>
        </div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Model Uptime</CardTitle>
              <Clock className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{health?.uptime ? formatUptime(health.uptime) : "N/A"}</div>
              <p className="text-xs text-muted-foreground">Continuous operation</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best Accuracy</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {analytics?.training_summary.best_accuracy
                  ? `${(analytics.training_summary.best_accuracy * 100).toFixed(1)}%`
                  : "N/A"}
              </div>
              <p className="text-xs text-muted-foreground">Peak model performance</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Trainings</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{analytics?.training_summary.total_trainings || 0}</div>
              <p className="text-xs text-muted-foreground">Training sessions completed</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">High Confidence</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{confidenceAnalysis?.prediction_patterns.high_confidence || 0}</div>
              <p className="text-xs text-muted-foreground">High confidence predictions</p>
            </CardContent>
          </Card>
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Performance Evolution Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Model Performance Evolution</CardTitle>
              <CardDescription>Training performance metrics over time</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip formatter={(value) => [`${value}%`, ""]} />
                  <Legend />
                  <Line type="monotone" dataKey="accuracy" stroke="#8884d8" strokeWidth={2} name="Accuracy" />
                  <Line type="monotone" dataKey="precision" stroke="#82ca9d" strokeWidth={2} name="Precision" />
                  <Line type="monotone" dataKey="recall" stroke="#ffc658" strokeWidth={2} name="Recall" />
                  <Line type="monotone" dataKey="f1_score" stroke="#ff7300" strokeWidth={2} name="F1 Score" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Data Distribution Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Training Data Distribution</CardTitle>
              <CardDescription>Distribution of training data by class</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={dataDistributionData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {dataDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Confidence Distribution Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Confidence Distribution</CardTitle>
              <CardDescription>Distribution of prediction confidence levels</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={confidenceDistributionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Threshold Analysis Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Threshold Analysis</CardTitle>
              <CardDescription>Accuracy vs confidence threshold</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={thresholdAnalysisData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="threshold" />
                  <YAxis domain={[0, 100]} />
                  <Tooltip formatter={(value) => [`${value}%`, "Accuracy"]} />
                  <Line type="monotone" dataKey="accuracy" stroke="#8884d8" strokeWidth={2} name="Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>

        {/* Confidence Analysis Summary */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Card>
            <CardHeader>
              <CardTitle>High Confidence</CardTitle>
              <CardDescription>Predictions with high confidence</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-green-600">
                {confidenceAnalysis?.confidence_distribution.high_confidence || 0}%
              </div>
              <p className="text-sm text-muted-foreground mt-2">Very reliable predictions</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Medium Confidence</CardTitle>
              <CardDescription>Predictions with medium confidence</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-yellow-600">
                {confidenceAnalysis?.confidence_distribution.medium_confidence || 0}%
              </div>
              <p className="text-sm text-muted-foreground mt-2">Moderately reliable predictions</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Low Confidence</CardTitle>
              <CardDescription>Predictions with low confidence</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold text-red-600">
                {confidenceAnalysis?.confidence_distribution.low_confidence || 0}%
              </div>
              <p className="text-sm text-muted-foreground mt-2">Consider model retraining</p>
            </CardContent>
          </Card>
        </div>

        {/* Training Summary */}
        <Card>
          <CardHeader>
            <CardTitle>Training Summary</CardTitle>
            <CardDescription>Overall model training statistics</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold mb-2">{analytics?.training_summary.total_models_trained || 0}</div>
                <p className="text-sm text-gray-600">Total Training Sessions</p>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold mb-2">
                  {analytics?.training_summary.latest_accuracy
                    ? `${(analytics.training_summary.latest_accuracy * 100).toFixed(1)}%`
                    : "N/A"}
                </div>
                <p className="text-sm text-gray-600">Latest Accuracy</p>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold mb-2">
                  {analytics?.training_summary.improvement
                    ? `${(analytics.training_summary.improvement * 100).toFixed(1)}%`
                    : "N/A"}
                </div>
                <p className="text-sm text-gray-600">Improvement</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  )
}
