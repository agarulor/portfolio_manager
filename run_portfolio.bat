@echo off
setlocal

cd /d "%~dp0"

echo === Ejecucion del programa ROBO UOC ADVISOR - TFG Alberto Garulo ===

REM creamos venv (solo si no existe)
if not exist ".venv\" (
    echo === Creando entorno virtual ===
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: No se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
)

REM activamos el venv
echo === Activando entorno virtual ===
call ".venv\Scripts\activate"
if errorlevel 1 (
    echo ERROR: No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

REM Actualizamos pip
echo === Actualizando pip ===
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: No se pudo actualizar pip.
    pause
    exit /b 1
)

REM instalamos los requerimientos
echo === Instalando requerimientos ===
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: No se pudieron instalar los requerimientos.
    pause
    exit /b 1
)

REM ejecutamos el código
echo === Ejecutando aplicacion ===
python -m streamlit run portfolio.py
if errorlevel 1 (
    echo ERROR: Fallo al ejecutar Streamlit.
    pause
    exit /b 1
)

echo === FIN del portfolio ===
pause
endlocal
