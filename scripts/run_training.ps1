# CarIntel — Run Distributed ML Training Pipeline
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "CarIntel: Training Spark MLlib Models (RF & GBT)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$sysPython = "C:\Users\$($env:USERNAME)\AppData\Local\Programs\Python\Python314\python.exe"
if (-not (Test-Path $sysPython)) {
    $candidate = Get-Command python.exe -CommandType Application | Where-Object { $_.Source -notmatch '\\\.venv\\' } | Select-Object -First 1
    if ($candidate) { $sysPython = $candidate.Source } else { $sysPython = "python" }
}

Write-Host "Training RandomForestRegressor..." -ForegroundColor Yellow
& $sysPython -m ml.train_rf

Write-Host "`nTraining GBTRegressor..." -ForegroundColor Yellow
& $sysPython -m ml.train_gbt

Write-Host "`nGenerating Unified Model Evaluation Report..." -ForegroundColor Yellow
& $sysPython -m ml.evaluate

Write-Host "`nML Pipeline Complete! Models saved to ml/models/" -ForegroundColor Green

