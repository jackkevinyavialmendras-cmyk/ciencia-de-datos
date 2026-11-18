@echo off
REM Doble clic para abrir la interfaz en el navegador (http://localhost:8501)
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Creando entorno e instalando librerias, la primera vez tarda unos minutos...
    python -m venv venv
    venv\Scripts\python.exe -m pip install -r requirements.txt
)
echo.
echo  Abriendo la interfaz en http://localhost:8501
echo  Para cerrarla, cierra esta ventana o presiona Ctrl+C.
echo.
venv\Scripts\python.exe -m streamlit run app\app.py --browser.gatherUsageStats false
pause
