# AWS Deployment Guide

This guide walks you through the steps to containerize your SLM and deploy it to AWS App Runner using Amazon Elastic Container Registry (ECR).

## Prerequisites
1. Ensure you have the [AWS CLI](https://aws.amazon.com/cli/) installed and configured (`aws configure`).
2. Ensure you have [Docker](https://www.docker.com/) installed and running.
3. **CRITICAL**: You must have run `src/prepare_dataset.py` and `src/train.py` so that `./models/brain-tumor-slm` is populated. The Docker build relies on this directory.

## Step 1: Create an ECR Repository
Create a repository to store your Docker image.
```bash
aws ecr create-repository --repository-name brain-tumor-slm-api --region us-east-1
```

## Step 2: Authenticate Docker with ECR
Retrieve an authentication token and authenticate your Docker client to your registry. Replace `<YOUR_AWS_ACCOUNT_ID>` with your actual AWS account ID.
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```

## Step 3: Build and Tag the Docker Image
Build your Docker image locally. Note: This will copy your pre-trained `./models` folder into the image.
```bash
docker build -t brain-tumor-slm-api .
docker tag brain-tumor-slm-api:latest <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/brain-tumor-slm-api:latest
```

## Step 4: Push the Image to ECR
```bash
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/brain-tumor-slm-api:latest
```

## Step 5: Deploy to AWS App Runner
1. Open the [AWS App Runner Console](https://console.aws.amazon.com/apprunner/).
2. Click **Create an App Runner service**.
3. Under **Repository type**, select **Container registry** and choose **Amazon ECR**.
4. Browse and select your `brain-tumor-slm-api:latest` image.
5. In **Service settings**, set the Port to `8080`.
6. **Important:** Allocate enough memory and CPU (Recommended: at least 2 vCPU and 4 GB RAM) for a PyTorch inference endpoint.
7. Review and create.

App Runner will deploy the container, and once it finishes, it will provide a default URL you can use to interact with your `/summarize` API endpoint.
