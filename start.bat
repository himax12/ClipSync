@echo off
REM Set PyTorch to allow loading trusted models (WhisperX/Pyannote)
REM This must be set BEFORE Python starts
SET TORCH_SERIALIZATION_UNSAFE_LEGACY_LOAD=1

REM Start backend and frontend in parallel
echo Starting Semantic A-Roll/B-Roll Engine...
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:8501
echo.

start "Backend API" cmd /k "cd /d %~dp0 && uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000"
timeout /t 5 /nobreak >nul
start "Frontend UI" cmd /k "cd /d %~dp0 && uv run streamlit run frontend/app.py"

echo.
echo ✓ Both servers starting...
echo   Press Ctrl+C in each window to stop
echo.
