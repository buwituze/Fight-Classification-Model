# Fighting 3D CNN Model API

This is a RESTful API for the Fighting 3D CNN Model, which is a deep learning model for detecting fighting in videos.

## Endpoints

- **POST /health**: Check the health of the API.
- **POST /model**: Get information about the model, including its version and supported video formats.
- **POST /predict**: Make a prediction on a given video file.
- **POST /retrain**: Retrain the model with new data.
- **GET /training-status**: Get the status of the training process.
- **DELETE /cancel-training**: Cancel the training process.
- **GET /analytics/dataset-stats**: Get statistics about the dataset.
- **POST /predict** : Predict Fight
- **GET /analytics/confidence-analysis**: Get Confidence Analysis
