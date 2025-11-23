# Quick Reference Card

## 🚀 Customer Churn Prediction - Deployment Commands

### Local Development

#### Setup & Run (Windows)
```powershell
# Interactive setup and run
.\run.ps1

# Manual steps
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

#### Setup & Run (Linux/Mac)
```bash
# Interactive setup and run
./run.sh

# Manual steps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

#### Train Model
```bash
python main.py
```

#### Test Streamlit Compatibility
```bash
# Run compatibility tests
python test_streamlit.py

# Should show: ✅ All tests passed! Your app is Streamlit-ready!
```

### Docker Commands

#### Quick Start
```bash
# Start everything
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f churn-prediction-app

# Stop everything
docker-compose down
```

#### Manual Docker
```bash
# Build image
docker build -t customer-churn-prediction .

# Run container
docker run -d -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --name churn-app \
  customer-churn-prediction

# View logs
docker logs -f churn-app

# Stop and remove
docker stop churn-app && docker rm churn-app
```

#### Training in Docker
```bash
# Run training separately
docker-compose --profile training up churn-training
```

### Kubernetes Commands

```bash
# Deploy to cluster
kubectl apply -f k8s-deployment.yaml

# Check deployment
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/customer-churn-prediction

# Scale manually
kubectl scale deployment customer-churn-prediction --replicas=5

# Delete deployment
kubectl delete -f k8s-deployment.yaml
```

### Health Checks

```bash
# Application health
curl http://localhost:8501/_stcore/health

# Python script
python health_check.py

# Docker container health
docker inspect --format='{{.State.Health.Status}}' churn-app
```

### Development Tasks (Makefile)

```bash
# View all commands
make help

# Install dependencies
make install

# Train model
make train

# Run app
make run

# Build Docker image
make docker-build

# Run with Docker
make docker-run

# Start with Docker Compose
make docker-compose

# Clean generated files
make clean

# Run tests
make test

# Lint code
make lint
```

### Cloud Deployment

#### Streamlit Cloud (Easiest)
```bash
# 1. Test locally first
python test_streamlit.py
streamlit run app.py

# 2. Push to GitHub
git add .
git commit -m "Deploy to Streamlit Cloud"
git push origin main

# 3. Go to share.streamlit.io
# - Sign in with GitHub
# - Select repository: customer-churn-prediction
# - Main file: app.py
# - Click Deploy!
```

See: [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md)

#### AWS ECS
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t customer-churn-prediction .
docker tag customer-churn-prediction:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/customer-churn-prediction:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/customer-churn-prediction:latest
```

#### Google Cloud Run
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/customer-churn-prediction
gcloud run deploy customer-churn-prediction \
  --image gcr.io/YOUR_PROJECT_ID/customer-churn-prediction \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501
```

#### Heroku
```bash
# Login and create app
heroku login
heroku create your-app-name

# Deploy
git push heroku main

# Scale
heroku ps:scale web=1
```

### Git Commands

```bash
# Initialize repository
git init
git add .
git commit -m "Initial commit - deployment ready"

# Add remote and push
git remote add origin https://github.com/username/repo.git
git push -u origin main
```

### Environment Management

```bash
# Copy environment template
cp .env.example .env

# For production
cp .env.production .env

# Edit environment
nano .env  # Linux/Mac
notepad .env  # Windows
```

### Monitoring & Debugging

```bash
# View application logs (local)
tail -f logs/*.log

# View Docker logs
docker logs -f churn-app

# View Kubernetes logs
kubectl logs -f deployment/customer-churn-prediction

# Docker container inspection
docker inspect churn-app

# Kubernetes pod inspection
kubectl describe pod <pod-name>

# Resource usage (Docker)
docker stats churn-app

# Resource usage (Kubernetes)
kubectl top pods
```

### Backup & Restore

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/ results/

# Restore models
tar -xzf models-backup-20231123.tar.gz

# Backup data
cp -r data/ data-backup-$(date +%Y%m%d)/
```

### Testing

```bash
# Test with sample prediction
curl -X POST http://localhost:8501/api/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 50.0}'

# Health check
curl http://localhost:8501/_stcore/health

# Load test (if you have apache bench)
ab -n 100 -c 10 http://localhost:8501/
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `.env` | Environment configuration |
| `docker-compose.yml` | Docker orchestration |
| `Dockerfile` | Container definition |
| `k8s-deployment.yaml` | Kubernetes manifests |
| `requirements.txt` | Python dependencies |
| `main.py` | Training pipeline |
| `app.py` | Streamlit application |

## 🌐 Access Points

- **Local**: http://localhost:8501
- **Docker**: http://localhost:8501
- **Production**: https://your-domain.com

## 📚 Documentation

- **Setup**: DEPLOYMENT_READY.md
- **Deploy**: DEPLOYMENT.md
- **Checklist**: DEPLOYMENT_CHECKLIST.md
- **Contribute**: CONTRIBUTING.md
- **Changes**: CHANGELOG.md

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8501 busy | `docker stop churn-app` or change port |
| Model not found | Run `python main.py` first |
| Permission denied | `chmod +x run.sh` (Linux/Mac) |
| Docker won't start | Check Docker daemon is running |
| Import errors | `pip install -r requirements.txt` |

## 📞 Get Help

- Check logs: `logs/` directory
- Review errors: No errors in VSCode
- Documentation: See markdown files in project root
- GitHub Issues: Open an issue

---

**Quick Start**: `docker-compose up -d` → http://localhost:8501

**Status**: ✅ Production Ready
