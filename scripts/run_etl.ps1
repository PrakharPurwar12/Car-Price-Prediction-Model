# CarIntel — Run Distributed ETL Pipeline
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "CarIntel: Executing Apache Spark Distributed ETL Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$sysPython = "C:\Users\$($env:USERNAME)\AppData\Local\Programs\Python\Python314\python.exe"
if (-not (Test-Path $sysPython)) {
    $candidate = Get-Command python.exe -CommandType Application | Where-Object { $_.Source -notmatch '\\\.venv\\' } | Select-Object -First 1
    if ($candidate) { $sysPython = $candidate.Source } else { $sysPython = "python" }
}

& $sysPython -m spark_processing.etl_pipeline
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nETL Pipeline Completed Successfully! Processed Parquet written to data/processed/vehicles.parquet" -ForegroundColor Green
    & $sysPython -m spark_processing.analytical_queries
} else {
    Write-Host "`nETL Pipeline Failed with Exit Code: $LASTEXITCODE" -ForegroundColor Red
}

