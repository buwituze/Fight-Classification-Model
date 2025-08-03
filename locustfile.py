from locust import HttpUser, task, between
import os

class FightAPITestUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(5)  # Higher weight - will run more often
    def test_health(self):
        """Test health endpoint"""
        self.client.get("/health")
    
    @task(3)
    def test_model_info(self):
        """Test model info endpoint"""
        self.client.get("/model-info")
    
    @task(2)
    def test_root(self):
        """Test root endpoint"""
        self.client.get("/")
    
    @task(2)
    def test_training_status(self):
        """Test training status endpoint"""
        self.client.get("/training-status")
    
    @task(1)
    def test_dataset_stats(self):
        """Test dataset stats endpoint"""
        self.client.get("/analytics/dataset-stats")
    
    @task(1)
    def test_confidence_analysis(self):
        """Test confidence analysis endpoint"""
        self.client.get("/analytics/confidence-analysis")
    
    @task(2)
    def test_predict_fight_video(self):
        """Test predict endpoint with fight video"""
        try:
            with open("test1.mp4", "rb") as f:
                files = {"file": ("test1.mp4", f, "video/mp4")}
                self.client.post("/predict", files=files)
        except FileNotFoundError:
            pass
    
    @task(2)
    def test_predict_normal_video(self):
        """Test predict endpoint with normal video"""
        try:
            with open("test.mp4", "rb") as f:
                files = {"file": ("test.mp4", f, "video/mp4")}
                self.client.post("/predict", files=files)
        except FileNotFoundError:
            pass
    