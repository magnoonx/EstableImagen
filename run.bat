@echo off
title Estudio de Modificacion de Imagen IA (Local)
cd /d "%~dp0"

echo ========================================================
echo   ESTUDIO DE MODIFICACION DE IMAGEN IA LOCAL
echo   100%% Privado, Local y Sin Censura
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No se encontro el entorno virtual en .venv.
    echo Asegurate de que el entorno este configurado.
    pause
    exit /b 1
)

echo Iniciando aplicacion web...
.venv\Scripts\python.exe app.py

pause

