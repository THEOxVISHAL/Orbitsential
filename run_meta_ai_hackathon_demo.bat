@echo off
title Lucknow Geospatial AI - Meta Developer Hackathon Edition
color 0B
echo ==============================================================================
echo   🛰️  LUCKNOW GEOSPATIAL AI : META AI DEVELOPER HACKATHON GRAND DEMO
echo   Autonomous Surveillance & Multi-Model Enforcement Platform
echo ==============================================================================
echo.
echo   META FOUNDATION STACK:
echo     [1] Meta SAM 2 (FAIR)       - Sub-Pixel Cadastral Boundary Delineation
echo     [2] Meta DINOv2 ViT (FAIR)  - Zero-Shot Structural Anomaly Heatmap
echo     [3] Meta FAISS (FAIR)       - Vector Similarity Legal Precedent Matching
echo     [4] Meta Llama 3.2 Vision   - Multi-Modal Satellite Forensic Investigator
echo     [5] Meta NLLB-200           - Hyper-Local Dialects (Awadhi ^& Bhojpuri)
echo     [6] Meta MMS Speech         - Tactical Police Wireless Audio Dispatch
echo     [7] Meta Llama Guard 3      - Statutory Compliance ^& Bias Safety Shield
echo.
echo ==============================================================================
echo.

cd /d "%~dp0"

echo [1/3] Verifying Python Environment ^& GPU Acceleration...
venv\Scripts\python.exe -c "import torch, faiss, dinov2; print('  ✓ PyTorch CUDA Active:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo.
echo [2/3] Executing Complete Meta AI Surveillance ^& Enforcement Pipeline...
echo       - Delineating sub-pixel boundaries with Meta SAM 2
echo       - Computing 384-d patch divergence heatmaps with Meta DINOv2
echo       - Retrieving historical court precedents via Meta FAISS
echo       - Drafting bilingual demolition orders with Meta Llama 3.2 Vision
echo       - Translating to Awadhi ^& Bhojpuri with Meta NLLB-200
echo       - Synthesizing walkie-talkie field audio with Meta MMS
echo       - Compiling official legal demolition warrants (PDF)...
venv\Scripts\python.exe run_enforcement_pipeline.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Enforcement pipeline encountered an error.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [3/3] Launching Live Command Center Dashboard (index.html)...
start "" "index.html"

echo.
echo ==============================================================================
echo   🚀 GRAND DEMO READY! 
echo   - Click "Meta AI Intelligence Dossier" on the dashboard
echo   - Inspect SAM 2 vs DINOv2 visual comparison
echo   - Play Meta MMS Tactical Radio Dispatch
echo   - Trigger 3D Tactical Drone Flyover
echo   - Download official Court Demolition Warrant (PDF)
echo ==============================================================================
echo.
timeout /t 6
