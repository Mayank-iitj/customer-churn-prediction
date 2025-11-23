# Deployment Guide

This guide provides comprehensive instructions for deploying the Customer Churn Prediction application in various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Environment Variables](#environment-variables)
- [Scaling and Production Considerations](#scaling-and-production-considerations)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying the application, ensure you have:

- Python 3.8 or higher
- Docker and Docker Compose (for containerized deployment)
- Git (for version control)
- Access to your data file (`customer_churn.csv`)

## Local Deployment

### Standard Python Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your configuration
   # (Use your favorite text editor)
   ```

5. **Place your data**:
   ```bash
   # Copy your customer churn dataset to data/
   cp /path/to/your/data.csv data/customer_churn.csv
   ```

6. **Train the model** (first time):
   ```bash
   python main.py
   ```

7. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

8. **Access the application**:
   - Open your browser to `http://localhost:8501`

### Using PowerShell Script (Windows)

```powershell
# Make the script executable and run
.\run.ps1
```

### Using Bash Script (Linux/Mac)

```bash
# Make the script executable
chmod +x run.sh

# Run the script
./run.sh
```

## Docker Deployment

Docker provides a consistent, portable deployment environment.

### Using Docker Compose (Recommended)

1. **Build and start the application**:
   ```bash
   docker-compose up -d
   ```

2. **View logs**:
   ```bash
   docker-compose logs -f churn-prediction-app
   ```

3. **Stop the application**:
   ```bash
   docker-compose down
   ```

4. **Train the model** (optional separate training):
   ```bash
   docker-compose --profile training up churn-training
   ```

### Using Docker Directly

1. **Build the Docker image**:
   ```bash
   docker build -t customer-churn-prediction .
   ```

2. **Run the container**:
   ```bash
   docker run -d \
     -p 8501:8501 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/results:/app/results \
     --name churn-app \
     customer-churn-prediction
   ```

3. **Check logs**:
   ```bash
   docker logs -f churn-app
   ```

4. **Stop the container**:
   ```bash
   docker stop churn-app
   docker rm churn-app
   ```

## Cloud Deployment

### AWS EC2 Deployment

1. **Launch an EC2 instance**:
   - Choose Ubuntu 20.04 LTS or Amazon Linux 2
   - Instance type: t2.medium or larger (recommended)
   - Configure security group to allow inbound traffic on port 8501

2. **Connect to your instance**:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Install Docker** (if using Docker):
   ```bash
   sudo apt update
   sudo apt install -y docker.io docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   ```

4. **Clone and deploy**:
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   docker-compose up -d
   ```

5. **Access your app**:
   - Navigate to `http://your-instance-ip:8501`

### AWS ECS/Fargate Deployment

1. **Push image to ECR**:
   ```bash
   # Authenticate to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

   # Build and tag
   docker build -t customer-churn-prediction .
   docker tag customer-churn-prediction:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/customer-churn-prediction:latest

   # Push
   docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/customer-churn-prediction:latest
   ```

2. **Create ECS task definition** with:
   - Container port: 8501
   - Memory: 2GB (recommended)
   - CPU: 1 vCPU
   - Mount EFS volumes for data persistence

3. **Create ECS service** and configure load balancer

### Heroku Deployment

1. **Install Heroku CLI**:
   ```bash
   # Follow instructions at https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**:
   ```bash
   heroku login
   ```

3. **Create app**:
   ```bash
   heroku create your-app-name
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

5. **Scale dynos**:
   ```bash
   heroku ps:scale web=1
   ```

### Google Cloud Run Deployment

1. **Build and push to Google Container Registry**:
   ```bash
   # Set project
   gcloud config set project YOUR_PROJECT_ID

   # Build
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/customer-churn-prediction

   # Deploy
   gcloud run deploy customer-churn-prediction \
     --image gcr.io/YOUR_PROJECT_ID/customer-churn-prediction \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --port 8501 \
     --memory 2Gi
   ```

### Azure Container Instances

1. **Login to Azure**:
   ```bash
   az login
   ```

2. **Create resource group**:
   ```bash
   az group create --name churn-prediction-rg --location eastus
   ```

3. **Create container instance**:
   ```bash
   az container create \
     --resource-group churn-prediction-rg \
     --name churn-prediction-app \
     --image customer-churn-prediction:latest \
     --dns-name-label churn-prediction \
     --ports 8501 \
     --cpu 2 \
     --memory 4
   ```

## Environment Variables

Key environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_PATH` | Path to the CSV data file | `data/customer_churn.csv` |
| `RANDOM_STATE` | Random seed for reproducibility | `42` |
| `SCALING_METHOD` | Feature scaling method | `standard` |
| `ENCODING_METHOD` | Categorical encoding method | `onehot` |
| `TUNE_HYPERPARAMETERS` | Enable hyperparameter tuning | `true` |
| `STREAMLIT_SERVER_PORT` | Port for Streamlit server | `8501` |

See `.env.example` for complete list.

## Scaling and Production Considerations

### Performance Optimization

1. **Model caching**: Models are cached in Streamlit using `@st.cache_resource`

2. **Batch prediction optimization**: Use batch prediction for large datasets

3. **Resource allocation**:
   - Minimum: 2GB RAM, 1 CPU
   - Recommended: 4GB RAM, 2 CPUs
   - For training: 8GB RAM, 4 CPUs

### Security Best Practices

1. **Environment variables**: Never commit `.env` files
2. **API keys**: Use secrets management (AWS Secrets Manager, Azure Key Vault)
3. **HTTPS**: Always use SSL/TLS in production
4. **Authentication**: Implement user authentication for production
5. **CORS**: Configure properly in Streamlit config

### Monitoring and Logging

1. **Application logs**: Check `logs/` directory
2. **Container logs**: Use `docker logs` or cloud provider logging
3. **Health checks**: Built-in health check endpoint at `/_stcore/health`
4. **Metrics**: Consider adding application metrics (Prometheus, DataDog)

### Data Persistence

1. **Volume mounts**: Ensure proper volume configuration for:
   - `data/` - Input datasets
   - `models/` - Trained models
   - `results/` - Training results
   - `logs/` - Application logs

2. **Backup strategy**: Regular backups of models and results

### Load Balancing

For high-traffic scenarios:

1. **Multiple instances**: Run multiple app instances behind a load balancer
2. **Session affinity**: Configure sticky sessions if needed
3. **Shared storage**: Use network storage (NFS, EFS, Azure Files) for models

## Troubleshooting

### Common Issues

**Issue**: Application fails to start
- **Solution**: Check logs for missing dependencies or configuration errors
- Verify all environment variables are set correctly

**Issue**: Model not found
- **Solution**: Train the model first using `python main.py`
- Check that model path in config matches actual location

**Issue**: Out of memory errors
- **Solution**: Increase container memory limits
- Reduce `SHAP_SAMPLE_SIZE` in configuration

**Issue**: Port already in use
- **Solution**: Change `STREAMLIT_SERVER_PORT` or stop conflicting service

**Issue**: Docker container won't start
- **Solution**: Check Docker logs: `docker logs <container-name>`
- Verify all volume paths exist

**Issue**: Cannot connect to application
- **Solution**: Check firewall/security group settings
- Verify correct port is exposed and mapped

### Health Checks

Test application health:

```bash
# Local
curl http://localhost:8501/_stcore/health

# Docker
docker exec churn-app curl http://localhost:8501/_stcore/health
```

Expected response: HTTP 200 OK

### Debug Mode

Enable debug logging:

1. Set `LOG_LEVEL=DEBUG` in `.env`
2. Check logs in `logs/` directory
3. Monitor Streamlit debug output

## Support

For issues and questions:
- Check the [README.md](README.md) for general usage
- Review logs in `logs/` directory
- Open an issue on GitHub

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [AWS ECS Guide](https://docs.aws.amazon.com/ecs/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/)

---

**Last Updated**: November 2025
