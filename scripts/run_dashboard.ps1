# CarIntel — Launch Interactive Streamlit Dashboard
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "CarIntel: Launching Streamlit Intelligence Dashboard" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$env:PYTHONPATH = "$PSScriptRoot\.."

# Windows Application Control blocks C-extension DLLs (.pyd) located on secondary drives (F:\.venv).
# Use system Python on C:\ to ensure complete AppControl compliance.
$sysPython = "C:\Users\$($env:USERNAME)\AppData\Local\Programs\Python\Python314\python.exe"
if (-not (Test-Path $sysPython)) {
    $candidate = Get-Command python.exe -CommandType Application | Where-Object { $_.Source -notmatch '\\\.venv\\' } | Select-Object -First 1
    if ($candidate) { $sysPython = $candidate.Source }
}

if ($sysPython -and (Test-Path $sysPython)) {
    Write-Host "Running with AppControl-compliant Python: $sysPython" -ForegroundColor Green
    & $sysPython -m streamlit run "$PSScriptRoot\..\dashboard\streamlit_app.py"
} else {
    python -m streamlit run "$PSScriptRoot\..\dashboard\streamlit_app.py"
}

