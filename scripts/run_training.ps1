# CarIntel — Run Distributed ML Training Pipeline
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "CarIntel: Training Spark MLlib Models (RF & GBT)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "Training RandomForestRegressor..." -ForegroundColor Yellow
python -m ml.train_rf

Write-Host "`nTraining GBTRegressor..." -ForegroundColor Yellow
python -m ml.train_gbt

Write-Host "`nGenerating Unified Model Evaluation Report..." -ForegroundColor Yellow
python -m ml.evaluate

Write-Host "`nML Pipeline Complete! Models saved to ml/models/" -ForegroundColor Green

