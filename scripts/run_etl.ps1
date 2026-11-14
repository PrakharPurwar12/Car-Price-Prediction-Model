# CarIntel — Run Distributed ETL Pipeline
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "CarIntel: Executing Apache Spark Distributed ETL Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

python -m spark_processing.etl_pipeline
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nETL Pipeline Completed Successfully! Processed Parquet written to data/processed/vehicles.parquet" -ForegroundColor Green
    python -m spark_processing.analytical_queries
} else {
    Write-Host "`nETL Pipeline Failed with Exit Code: $LASTEXITCODE" -ForegroundColor Red
}

