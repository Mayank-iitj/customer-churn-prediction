# Makefile for Customer Churn Prediction

.PHONY: help install train run docker-build docker-run docker-stop clean test lint

help:
	@echo "Customer Churn Prediction - Available Commands"
	@echo "==============================================="
	@echo "install        - Install dependencies in virtual environment"
	@echo "train          - Train the machine learning model"
	@echo "run            - Run the Streamlit application"
	@echo "docker-build   - Build Docker image"
	@echo "docker-run     - Run Docker container"
	@echo "docker-stop    - Stop Docker container"
	@echo "docker-compose - Run with Docker Compose"
	@echo "clean          - Clean generated files and caches"
	@echo "test           - Run tests"
	@echo "lint           - Lint code with flake8"

install:
	@echo "Creating virtual environment and installing dependencies..."
	python -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Installation complete!"

train:
	@echo "Training model..."
	python main.py

run:
	@echo "Starting Streamlit app..."
	@echo "Access at http://localhost:8501"
	streamlit run app.py

docker-build:
	@echo "Building Docker image..."
	docker build -t customer-churn-prediction:latest .

docker-run: docker-build
	@echo "Running Docker container..."
	docker run -d \
		-p 8501:8501 \
		-v $$(pwd)/data:/app/data \
		-v $$(pwd)/models:/app/models \
		-v $$(pwd)/results:/app/results \
		-v $$(pwd)/logs:/app/logs \
		--name churn-app \
		customer-churn-prediction:latest
	@echo "Container started! Access at http://localhost:8501"

docker-stop:
	@echo "Stopping Docker container..."
	docker stop churn-app || true
	docker rm churn-app || true

docker-compose:
	@echo "Starting with Docker Compose..."
	docker-compose up -d
	@echo "Services started! Access at http://localhost:8501"

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov 2>/dev/null || true
	@echo "Cleanup complete!"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html

lint:
	@echo "Linting code..."
	flake8 src/ --max-line-length=127 --exclude=__pycache__
	@echo "Linting complete!"
