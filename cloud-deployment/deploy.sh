#!/bin/bash

# MET Art Display Predictor - Google Cloud Run Deployment Script
# This script builds and deploys your containerized API to Google Cloud Run

set -e  # Exit on error

# Configuration
PROJECT_ID="your-gcp-project-id"  # UPDATE THIS
SERVICE_NAME="met-art-predictor"
REGION="us-central1"  # or your preferred region
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "======================================"
echo "MET Art Predictor - Cloud Run Deployment"
echo "======================================"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if model file exists
if [ ! -f "model.cbm" ]; then
    echo "❌ Error: model.cbm not found"
    echo "Please copy your trained model to this directory:"
    echo "  cp ../notebooks/models/catboost_model_optimized_parameters_PRAUC_1000iter_e5large.cbm model.cbm"
    exit 1
fi

echo "✓ Model file found"
echo ""

# Authenticate (if needed)
echo "Checking authentication..."
gcloud auth print-access-token > /dev/null 2>&1 || gcloud auth login

# Set project
echo "Setting project to: ${PROJECT_ID}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo ""
echo "Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

# Build the container
echo ""
echo "Building container image..."
echo "This may take 5-10 minutes..."
gcloud builds submit --tag ${IMAGE_NAME}

# Deploy to Cloud Run
echo ""
echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 1 \
    --min-instances 0 \
    --cpu-boost \
    --concurrency 10 \
    --timeout 60s

# Get the service URL
echo ""
echo "======================================"
echo "✓ Deployment complete!"
echo "======================================"
echo ""
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test the API:"
echo "  curl ${SERVICE_URL}/health"
echo ""
echo "API Documentation:"
echo "  ${SERVICE_URL}/docs"
echo ""
echo "Cost estimates (with current settings):"
echo "  - First 2M requests/month: FREE"
echo "  - Additional requests: ~$0.40 per 1M"
echo "  - Memory/CPU: Only charged when handling requests"
echo "  - Scales to zero when idle: $0 cost"
echo ""

