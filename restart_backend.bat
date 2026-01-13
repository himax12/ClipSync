@echo off
echo Clearing Python cache...
for /d /r "backend" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q "backend\*.pyc" 2>nul

echo.
echo ================================
echo Python cache cleared!
echo.
echo Now restart backend with:
echo   uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo ================================
pause
