#!/bin/bash

echo "=== Ejecucion del programa ROBO UOC ADVISOR - TFG Alberto Garulo ==="

# Aseguramos que el script está en el directorio
cd "$(dirname "$0")" || exit 1

# comprobamos Python
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: Python3 no esta instalado."
    echo "Instalar con: sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Creamos el entorno virtual, si no existe
if [ ! -d ".venv" ]; then
    echo "=== Creando entorno virtual ==="
    python3 -m venv .venv || {
        echo "ERROR: Error a la hora de crear el entorno."
        echo "On Debian/Ubuntu run:"
        echo "  sudo apt install python3-venv"
        exit 1
    }
fi

# Activamos el entorno virtual
echo "=== Activando entorno virtual ==="
source .venv/bin/activate

# Actualizamos pip
echo "=== Actualizando pip ==="
python -m pip install --upgrade pip

# Instalar requerimientos
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: no se encontró requirements.txt"
    exit 1
fi

echo "=== Instalando requerimientos ==="
pip install -r requirements.txt

# Run Streamlit app
echo "=== Ejecutando aplicacion ==="
streamlit run portfolio.py

echo "=== Aplicación finalizada ==="
