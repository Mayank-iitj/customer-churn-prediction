#!/bin/bash

# Customer Churn Prediction - Startup Script
# This script sets up and runs the application

set -e

echo "=========================================="
echo "Customer Churn Prediction - Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

print_info "Python version: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_info "Virtual environment created."
else
    print_info "Virtual environment already exists."
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
print_info "Installing dependencies..."
pip install -r requirements.txt --quiet

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found."
    if [ -f ".env.example" ]; then
        print_info "Copying .env.example to .env..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration."
    else
        print_error ".env.example not found. Please create .env file manually."
    fi
fi

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p data logs models results notebooks

# Check if data file exists
if [ ! -f "data/customer_churn.csv" ]; then
    print_warning "Data file not found at data/customer_churn.csv"
    print_warning "Please place your dataset in the data/ directory before training."
fi

# Ask user what to do
echo ""
echo "What would you like to do?"
echo "1) Train model (python main.py)"
echo "2) Run Streamlit app (streamlit run app.py)"
echo "3) Both (train then run app)"
echo "4) Exit"
read -p "Enter your choice [1-4]: " choice

case $choice in
    1)
        print_info "Starting model training..."
        python main.py
        print_info "Training complete!"
        ;;
    2)
        print_info "Starting Streamlit app..."
        print_info "Access the app at http://localhost:8501"
        streamlit run app.py
        ;;
    3)
        print_info "Starting model training..."
        python main.py
        print_info "Training complete!"
        print_info "Starting Streamlit app..."
        print_info "Access the app at http://localhost:8501"
        streamlit run app.py
        ;;
    4)
        print_info "Exiting..."
        exit 0
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac
