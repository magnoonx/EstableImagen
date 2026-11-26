@echo off
title Instalar Soporte GPU NVIDIA (CUDA)
cd /d "%~dp0"

echo ========================================================
echo   ACTIVACION DE ACELERACION POR GPU NVIDIA (CUDA)
echo ========================================================
echo.
echo Este script instalara los controladores de PyTorch con soporte
echo para tarjetas graficas dedicadas NVIDIA (GeForce RTX / GTX).
echo.
echo Descargando e instalando PyTorch CUDA...
.venv\Scripts\pip.exe install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
echo.
echo ========================================================
echo Proceso finalizado. Puedes iniciar la aplicacion con run.bat
echo ========================================================
pause
