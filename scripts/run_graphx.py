"""
scripts/run_graphx.py
Executes the Scala Apache Spark GraphX pipeline for CarIntel:
Runs GraphMetricsExporter to compute PageRank, Connected Components, and Degree Centrality,
and exports graph_nodes.parquet and graph_summary.json.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SPARK_HOME = Path(os.environ.get("SPARK_HOME", r"C:\spark\spark-4.2.0-bin-hadoop3"))
JAVA_HOME = Path(os.environ.get("JAVA_HOME", r"C:\Program Files\Java\jdk-17"))

SPARK_JARS_DIR = SPARK_HOME / "jars"
WIN_FS_JAR = ROOT_DIR / "config" / "win_local_fs.jar"
GRAPHX_JAR = ROOT_DIR / "graphx" / "target" / "carintel-graphx.jar"

INPUT_PARQUET = ROOT_DIR / "data" / "processed" / "vehicles.parquet"
OUTPUT_PARQUET = ROOT_DIR / "data" / "graph_metrics" / "graph_nodes.parquet"
OUTPUT_JSON = ROOT_DIR / "data" / "graph_metrics" / "graph_summary.json"


def run_graphx():
    if not GRAPHX_JAR.exists():
        print(f"Error: {GRAPHX_JAR} does not exist. Run scripts/compile_graphx.py first.")
        return False

    print("Executing Scala Spark GraphX Analytics Pipeline...")
    print(f"  Input:  {INPUT_PARQUET}")
    print(f"  Output Parquet: {OUTPUT_PARQUET}")
    print(f"  Output JSON:    {OUTPUT_JSON}")

    java_bin = str(JAVA_HOME / "bin" / "java.exe")
    classpath = f"{GRAPHX_JAR};{WIN_FS_JAR};{SPARK_JARS_DIR}\\*"

    jvm_options = [
        "-Xmx3g",
        "-Dspark.master=local[4]",
        "-Dspark.driver.host=127.0.0.1",
        "-Dspark.driver.bindAddress=127.0.0.1",
        "-Dspark.hadoop.fs.file.impl=com.carintel.hadoop.WinLocalFileSystem",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    ]

    cmd = [
        java_bin,
    ] + jvm_options + [
        "-cp",
        classpath,
        "com.carintel.graph.GraphMetricsExporter",
        str(INPUT_PARQUET),
        str(OUTPUT_PARQUET),
        str(OUTPUT_JSON),
    ]

    res = subprocess.run(cmd, text=True)
    return res.returncode == 0


if __name__ == "__main__":
    success = run_graphx()
    if not success:
        sys.exit(1)

