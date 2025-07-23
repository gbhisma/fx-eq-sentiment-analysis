# Change this to your virtual environment folder if different
$venvPath = ".\.venv\Scripts\activate"

# Check if Ollama is running
$ollamaRunning = (Get-Process -Name "ollama" -ErrorAction SilentlyContinue)

if (-not $ollamaRunning) {
    Write-Host "Starting Ollama..."
    Start-Process -NoNewWindow -FilePath "ollama" -ArgumentList "run llama3"
    Start-Sleep -Seconds 5
} else {
    Write-Host "Ollama is already running."
}

# Activate virtual environment and launch Streamlit
Write-Host "Activating virtual environment..."
& $venvPath

Write-Host "Launching Streamlit app..."
streamlit run main.py
