# ClipSync Deployment Guide

## Google Cloud Setup

### 1. Enable Required APIs

Run these commands in Google Cloud Shell or gcloud CLI:

```bash
# Set project
gcloud config set project ai-automation-466002

# Enable APIs
gcloud services enable \
  aiplatform.googleapis.com \
  speech.googleapis.com \
  generativelanguage.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com
```

### 2. Create Artifact Registry (for Docker images)

```bash
gcloud artifacts repositories create clipsync \
  --repository-format=docker \
  --location=us-central1
```

---

## Backend Deployment (Cloud Run)

### 1. Build and Push Docker Image

```bash
cd backend

# Authenticate
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build
docker build -t us-central1-docker.pkg.dev/ai-automation-466002/clipsync/backend:latest .

# Push
docker push us-central1-docker.pkg.dev/ai-automation-466002/clipsync/backend:latest
```

### 2. Deploy to Cloud Run

```bash
gcloud run deploy clipsync-backend \
  --image=us-central1-docker.pkg.dev/ai-automation-466002/clipsync/backend:latest \
  --platform=managed \
  --region=us-central1 \
  --allow-unauthenticated \
  --port=8000 \
  --memory=2Gi \
  --cpu=2 \
  --set-env-vars="GCP_PROJECT_ID=ai-automation-466002,GCP_LOCATION=us-central1"
```

**Important:** Set `GOOGLE_API_KEY` in Cloud Run environment variables after deployment.

---

## Frontend Deployment (Vercel)

### 1. Connect to Vercel

```bash
cd frontend-next
npx vercel
```

### 2. Deploy with Domain

```bash
# Production deployment
npx vercel --prod

# Set domain
npx vercel domains add clipsync.me
npx vercel alias clipsync.me
```

### 3. Update Backend CORS

After getting Vercel URL, update backend `.env`:
```
CORS_ORIGINS=https://clipsync.me,https://www.clipsync.me,http://localhost:3000
```

Then redeploy backend:
```bash
gcloud run deploy clipsync-backend \
  --image=us-central1-docker.pkg.dev/ai-automation-466002/clipsync/backend:latest \
  --platform=managed \
  --region=us-central1 \
  --set-env-vARS="GCP_PROJECT_ID=ai-automation-466002,GCP_LOCATION=us-central1,CORS_ORIGINS=https://clipsync.me,https://www.clipsync.me,http://localhost:3000"
```

---

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend-next
npm install
npm run dev
```

---

## Environment Variables

### Backend (.env)
```
GCP_PROJECT_ID=ai-automation-466002
GCP_LOCATION=us-central1
GOOGLE_API_KEY=your_api_key_here
CORS_ORIGINS=https://clipsync.me,https://www.clipsync.me,http://localhost:3000
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
DATA_DIR=./data
```

### Frontend
Create `.env.local`:
```
NEXT_PUBLIC_API_URL=https://your-cloudrun-url.run.app
```
