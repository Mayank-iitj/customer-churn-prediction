# Deployment Checklist

Use this checklist to ensure a smooth deployment of the Customer Churn Prediction application.

## Pre-Deployment

### Code & Configuration
- [ ] All code changes committed and pushed to repository
- [ ] `.env` file configured with production values (do NOT commit this)
- [ ] Environment variables validated against `.env.example`
- [ ] All sensitive data removed from code
- [ ] Configuration reviewed in `src/config.py`
- [ ] Dependencies up to date in `requirements.txt`

### Data & Models
- [ ] Training data available and accessible
- [ ] Model trained and validated (run `python main.py`)
- [ ] Model files present in `models/` or `results/` directory
- [ ] Model performance meets acceptance criteria
- [ ] Preprocessor saved alongside model

### Testing
- [ ] All unit tests passing (if available)
- [ ] Application tested locally
- [ ] Streamlit app runs without errors
- [ ] Predictions work correctly
- [ ] Batch prediction tested with sample data
- [ ] Error handling verified

### Documentation
- [ ] README.md reviewed and updated
- [ ] DEPLOYMENT.md reviewed
- [ ] Environment variables documented
- [ ] API endpoints documented (if applicable)

## Docker Deployment

### Build & Test
- [ ] Dockerfile reviewed
- [ ] Docker image builds successfully: `docker build -t customer-churn-prediction .`
- [ ] Container runs locally: `docker run -p 8501:8501 customer-churn-prediction`
- [ ] Health check works: `curl http://localhost:8501/_stcore/health`
- [ ] Volume mounts configured correctly
- [ ] Container logs reviewed for errors

### Docker Compose
- [ ] `docker-compose.yml` reviewed
- [ ] Services start successfully: `docker-compose up`
- [ ] All volumes mounted correctly
- [ ] Network configuration verified
- [ ] Services communicate as expected

### Container Registry
- [ ] Container registry access configured (Docker Hub, ECR, GCR, ACR)
- [ ] Image tagged appropriately
- [ ] Image pushed to registry
- [ ] Image pull tested from registry

## Cloud Deployment

### AWS
- [ ] AWS credentials configured
- [ ] IAM roles and policies set up
- [ ] Security groups configured (port 8501 open)
- [ ] ECS task definition created (if using ECS)
- [ ] Load balancer configured (if needed)
- [ ] EFS/EBS volumes for persistence (if needed)
- [ ] CloudWatch logging configured

### Google Cloud
- [ ] GCP project created and selected
- [ ] Service account with appropriate permissions
- [ ] Cloud Run service configured (if using)
- [ ] Persistent storage configured (if needed)
- [ ] Cloud Logging enabled
- [ ] Domain/DNS configured (if needed)

### Azure
- [ ] Azure subscription active
- [ ] Resource group created
- [ ] Container instance/App Service configured
- [ ] Storage account for persistence (if needed)
- [ ] Application Insights configured (optional)
- [ ] Domain/DNS configured (if needed)

### Kubernetes
- [ ] Kubernetes cluster available
- [ ] `kubectl` configured and authenticated
- [ ] Persistent volumes created
- [ ] Secrets created for sensitive data
- [ ] ConfigMaps created for configuration
- [ ] Deployment manifest applied: `kubectl apply -f k8s-deployment.yaml`
- [ ] Service exposed and accessible
- [ ] Ingress configured (if needed)
- [ ] Auto-scaling configured and tested

## Security

### Application Security
- [ ] HTTPS/SSL configured
- [ ] Authentication implemented (if required)
- [ ] CORS settings configured correctly
- [ ] XSRF protection enabled
- [ ] Input validation in place
- [ ] No sensitive data in logs

### Infrastructure Security
- [ ] Firewall rules configured
- [ ] Network security groups set up
- [ ] Secrets managed securely (AWS Secrets Manager, etc.)
- [ ] No hardcoded credentials in code
- [ ] Container image scanned for vulnerabilities
- [ ] Regular security updates planned

### Data Security
- [ ] Data encryption at rest (if required)
- [ ] Data encryption in transit (HTTPS)
- [ ] Access controls on data storage
- [ ] Backup strategy for models and data
- [ ] Data retention policy defined

## Monitoring & Logging

### Application Monitoring
- [ ] Health checks configured
- [ ] Application logs accessible
- [ ] Error tracking set up (Sentry, etc.)
- [ ] Performance monitoring configured
- [ ] Alerts configured for errors/downtime

### Infrastructure Monitoring
- [ ] CPU/Memory metrics monitored
- [ ] Disk space monitored
- [ ] Network traffic monitored
- [ ] Container/pod health monitored
- [ ] Auto-restart policies configured

### Logging
- [ ] Log aggregation configured
- [ ] Log retention policy set
- [ ] Sensitive data excluded from logs
- [ ] Log levels appropriate for environment
- [ ] Centralized logging system (optional)

## Performance & Scaling

### Performance
- [ ] Resource limits configured (CPU, memory)
- [ ] Application performance tested under load
- [ ] Response times acceptable
- [ ] Caching configured where appropriate
- [ ] Database connection pooling (if using DB)

### Scaling
- [ ] Horizontal scaling configured (if needed)
- [ ] Load balancer set up (if multiple instances)
- [ ] Auto-scaling policies configured
- [ ] Session management for multiple instances
- [ ] Shared storage for models across instances

## Post-Deployment

### Verification
- [ ] Application accessible at production URL
- [ ] All features working as expected
- [ ] Single prediction tested
- [ ] Batch prediction tested
- [ ] Model information displays correctly
- [ ] Logs being generated properly

### Documentation
- [ ] Deployment documented with specifics
- [ ] Access URLs documented
- [ ] Credentials stored securely
- [ ] Runbook created for operations team
- [ ] Known issues documented

### Communication
- [ ] Stakeholders notified of deployment
- [ ] Access provided to authorized users
- [ ] Support team briefed
- [ ] Monitoring dashboard shared
- [ ] Escalation process defined

## Rollback Plan

### Preparation
- [ ] Previous version image/deployment saved
- [ ] Rollback procedure documented
- [ ] Database migration rollback plan (if applicable)
- [ ] Rollback tested in staging environment

### Execution (if needed)
- [ ] Stop current deployment
- [ ] Restore previous version
- [ ] Verify rollback successful
- [ ] Notify stakeholders
- [ ] Document rollback reason

## Maintenance

### Regular Tasks
- [ ] Update schedule defined
- [ ] Backup schedule configured
- [ ] Log rotation configured
- [ ] Certificate renewal process (if using HTTPS)
- [ ] Dependency update process defined

### Monitoring Schedule
- [ ] Daily: Health checks, error logs
- [ ] Weekly: Performance metrics, resource usage
- [ ] Monthly: Security updates, dependency updates
- [ ] Quarterly: Model retraining evaluation

## Sign-Off

- [ ] Technical lead approval
- [ ] Security review completed
- [ ] Operations team ready
- [ ] Deployment date/time scheduled
- [ ] Stakeholders informed

---

**Deployment Date**: _______________

**Deployed By**: _______________

**Version**: _______________

**Environment**: _______________

**Notes**:
_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

---

## Quick Commands Reference

```bash
# Build Docker image
docker build -t customer-churn-prediction .

# Run locally
docker run -p 8501:8501 customer-churn-prediction

# Docker Compose
docker-compose up -d

# Kubernetes
kubectl apply -f k8s-deployment.yaml

# Check health
curl http://localhost:8501/_stcore/health

# View logs (Docker)
docker logs -f churn-app

# View logs (Kubernetes)
kubectl logs -f deployment/customer-churn-prediction
```
