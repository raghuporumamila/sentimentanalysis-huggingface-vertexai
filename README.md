# sentimentanalysis-huggingface-vertexai
This code demo shows how we can deploy hugging face sentiment analysis model to Google Vertex AI

Here's a step-by-step guide to using this code:

1. Prerequisites:
```bash
pip install google-cloud-aiplatform transformers
```

2. Authentication:
- Set up Google Cloud credentials
- Enable Vertex AI API
- Set up appropriate IAM permissions

3. Implementation Steps:

a. Model Deployment:
- Package your model
- Upload to Google Cloud Storage
- Deploy using Vertex AI endpoints

b. Making Predictions:
- Use the deployed endpoint
- Send requests in the correct format
- Process responses as needed
