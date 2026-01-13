# Setup Script for RoadBuddy Winning Architecture

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "="*69 -ForegroundColor Cyan
Write-Host "RoadBuddy Winning Architecture - Setup" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "="*69 -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Gray

if ($pythonVersion -match "Python 3\.(1[0-9]|[0-9]{2,})\.") {
    Write-Host "  ✓ Python version OK" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python 3.10+ required!" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists" -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/6] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

# Upgrade pip
Write-Host ""
Write-Host "[4/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet
Write-Host "  ✓ pip upgraded" -ForegroundColor Green

# Install requirements
Write-Host ""
Write-Host "[5/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take 5-10 minutes..." -ForegroundColor Gray
pip install -r requirements.txt --quiet
Write-Host "  ✓ Dependencies installed" -ForegroundColor Green

# Create necessary directories
Write-Host ""
Write-Host "[6/6] Creating directories..." -ForegroundColor Yellow
$dirs = @(
    "data\raw\train",
    "data\raw\public_test",
    "data\raw\private_test",
    "data\processed",
    "data\knowledge_base",
    "checkpoints",
    "outputs",
    "logs"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Host "  ✓ Directories created" -ForegroundColor Green

# Generate knowledge base
Write-Host ""
Write-Host "Generating default knowledge base..." -ForegroundColor Yellow
python -c "
from src.experts.knowledge_base import TrafficKnowledgeBase
from omegaconf import OmegaConf

config = OmegaConf.load('configs/config.yaml')
kb = TrafficKnowledgeBase(config)
kb.save_database()
print('  ✓ Knowledge base saved')
"

# Summary
Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "="*69 -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "="*69 -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Place your data in data/raw/" -ForegroundColor Gray
Write-Host "  2. Download YOLOv10 weights to checkpoints/" -ForegroundColor Gray
Write-Host "  3. Run inference: python scripts\inference.py --test_json <path>" -ForegroundColor Gray
Write-Host ""
Write-Host "For more information, see README.md" -ForegroundColor Gray
Write-Host ""
