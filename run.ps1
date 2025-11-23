# Customer Churn Prediction - Startup Script (PowerShell)
# This script sets up and runs the application

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Customer Churn Prediction - Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Print-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Print-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Print-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Print-Info "Python version: $pythonVersion"
} catch {
    Print-Error "Python is not installed. Please install Python 3.8 or higher."
    exit 1
}

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Print-Info "Creating virtual environment..."
    python -m venv venv
    Print-Info "Virtual environment created."
} else {
    Print-Info "Virtual environment already exists."
}

# Activate virtual environment
Print-Info "Activating virtual environment..."
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Print-Info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
Print-Info "Installing dependencies..."
pip install -r requirements.txt --quiet

# Check if .env file exists
if (-Not (Test-Path ".env")) {
    Print-Warning ".env file not found."
    if (Test-Path ".env.example") {
        Print-Info "Copying .env.example to .env..."
        Copy-Item .env.example .env
        Print-Warning "Please edit .env file with your configuration."
    } else {
        Print-Error ".env.example not found. Please create .env file manually."
    }
}

# Create necessary directories
Print-Info "Creating necessary directories..."
$directories = @("data", "logs", "models", "results", "notebooks")
foreach ($dir in $directories) {
    if (-Not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Check if data file exists
if (-Not (Test-Path "data\customer_churn.csv")) {
    Print-Warning "Data file not found at data\customer_churn.csv"
    Print-Warning "Please place your dataset in the data\ directory before training."
}

# Ask user what to do
Write-Host ""
Write-Host "What would you like to do?" -ForegroundColor Cyan
Write-Host "1) Train model (python main.py)"
Write-Host "2) Run Streamlit app (streamlit run app.py)"
Write-Host "3) Both (train then run app)"
Write-Host "4) Exit"
Write-Host ""

$choice = Read-Host "Enter your choice [1-4]"

switch ($choice) {
    "1" {
        Print-Info "Starting model training..."
        python main.py
        Print-Info "Training complete!"
    }
    "2" {
        Print-Info "Starting Streamlit app..."
        Print-Info "Access the app at http://localhost:8501"
        streamlit run app.py
    }
    "3" {
        Print-Info "Starting model training..."
        python main.py
        Print-Info "Training complete!"
        Print-Info "Starting Streamlit app..."
        Print-Info "Access the app at http://localhost:8501"
        streamlit run app.py
    }
    "4" {
        Print-Info "Exiting..."
        exit 0
    }
    default {
        Print-Error "Invalid choice. Exiting."
        exit 1
    }
}
