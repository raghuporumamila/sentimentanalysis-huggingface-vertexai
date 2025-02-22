import os
from google.cloud import aiplatform
from google.cloud.aiplatform import Model
from transformers import pipeline

def setup_vertex_ai(project_id, region):
    """Initialize Vertex AI with project and region."""
    aiplatform.init(project=project_id, location=region)

def deploy_model_vertex(
    project_id: str,
    region: str,
    model_display_name: str,
    container_image_uri: str,
    artifact_uri: str
):
    """Deploy a Hugging Face model to Vertex AI."""
    
    model = Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri=container_image_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health"
    )
    
    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
        sync=True
    )
    
    return endpoint

def predict_vertex(endpoint, instances):
    """Make predictions using deployed endpoint."""
    response = endpoint.predict(instances=instances)
    return response.predictions

# Deployment and prediction
def main():
    PROJECT_ID = "your-project-id"
    REGION = "us-central1"
    MODEL_DISPLAY_NAME = "huggingface-sentiment"
    CONTAINER_IMAGE_URI = "gcr.io/your-project/huggingface-sentiment:v1"
    ARTIFACT_URI = "gs://your-bucket/model-artifacts"
    
    # Initialize Vertex AI
    setup_vertex_ai(PROJECT_ID, REGION)
    
    # Deploy model
    endpoint = deploy_model_vertex(
        PROJECT_ID,
        REGION,
        MODEL_DISPLAY_NAME,
        CONTAINER_IMAGE_URI,
        ARTIFACT_URI
    )
    
    # Make predictions
    test_instances = [
        {"text": "This is amazing!"},
        {"text": "This could be better."}
    ]
    predictions = predict_vertex(endpoint, test_instances)
    print("Predictions:", predictions)
