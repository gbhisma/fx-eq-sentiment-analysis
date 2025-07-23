@echo off
SETLOCAL

REM === Set your virtual environment activation path ===
SET VENV_PATH=.venv\Scripts\activate.bat

REM === Check if Ollama is already running ===
tasklist /FI "IMAGENAME eq ollama.exe" | find /I "ollama.exe" >nul
IF ERRORLEVEL 1 (
    echo Ollama is not running. Starting Ollama with Llama3...
    start "" /MIN cmd /c "ollama run llama3"
    timeout /t 5 >nul
) ELSE (
    echo Ollama is already running.
)

REM === Activate virtual environment ===
CALL %VENV_PATH%

REM === Run the Streamlit app ===
streamlit run main.py

ENDLOCAL
