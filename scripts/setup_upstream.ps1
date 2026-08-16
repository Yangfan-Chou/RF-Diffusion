@echo off
REM Setup script for Windows to download upstream RF-Diffusion

set UPSTREAM_DIR=%~dp0upstream\RF-Diffusion

echo === RF-Diffusion Setup Script ===
echo Upstream directory: %UPSTREAM_DIR%
echo.

if exist "%UPSTREAM_DIR%" (
    echo Upstream RF-Diffusion already exists
) else (
    echo Cloning upstream RF-Diffusion...
    git clone https://github.com/mobicom24/RF-Diffusion.git "%UPSTREAM_DIR%"
)

cd /d "%UPSTREAM_DIR%"

echo.
echo Downloading dataset and model weights...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/mobicom24/RF-Diffusion/releases/download/dataset_model/dataset.zip' -OutFile 'dataset.zip'"
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/mobicom24/RF-Diffusion/releases/download/dataset_model/model.zip' -OutFile 'model.zip'"

echo Extracting archives...
powershell -Command "Expand-Archive -Path dataset.zip -DestinationPath '.' -Force"
powershell -Command "Expand-Archive -Path model.zip -DestinationPath '.' -Force"

echo Restructuring directories...
if not exist "dataset" mkdir dataset
move wifi dataset\ 2>nul
move fmcw dataset\ 2>nul
move mimo dataset\ 2>nul

REM Cleanup
del dataset.zip
del model.zip

echo.
echo === Setup Complete ===
