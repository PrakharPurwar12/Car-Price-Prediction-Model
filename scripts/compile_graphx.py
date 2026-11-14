"""
scripts/compile_graphx.py
Compiles CarIntel Scala GraphX code using the embedded Scala compiler within Spark jars
and packages it into graphx/target/carintel-graphx.jar.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SPARK_HOME = Path(os.environ.get("SPARK_HOME", r"C:\spark\spark-4.2.0-bin-hadoop3"))
JAVA_HOME = Path(os.environ.get("JAVA_HOME", r"C:\Program Files\Java\jdk-17"))

SPARK_JARS_DIR = SPARK_HOME / "jars"
SCALA_SRC_DIR = ROOT_DIR / "graphx" / "src" / "main" / "scala" / "com" / "carintel" / "graph"
TARGET_CLASSES_DIR = ROOT_DIR / "graphx" / "target" / "classes"
OUTPUT_JAR = ROOT_DIR / "graphx" / "target" / "carintel-graphx.jar"
WIN_FS_JAR = ROOT_DIR / "config" / "win_local_fs.jar"


def compile_scala():
    TARGET_CLASSES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JAR.parent.mkdir(parents=True, exist_ok=True)

    scala_files = list(SCALA_SRC_DIR.glob("*.scala"))
    if not scala_files:
        print("No Scala files found!")
        return False

    print(f"Compiling {len(scala_files)} Scala GraphX files...")
    for sf in scala_files:
        print(f"  - {sf.name}")

    # Use Java classpath wildcard to avoid Windows 8191-char command line limit
    classpath = f"{WIN_FS_JAR};{SPARK_JARS_DIR}\\*"

    # JDK 17 JVM module opens
    jvm_opens = [
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

    java_bin = str(JAVA_HOME / "bin" / "java.exe")
    jar_bin = str(JAVA_HOME / "bin" / "jar.exe")

    compile_cmd = [
        java_bin,
        "-Xmx2g",
    ] + jvm_opens + [
        "-cp",
        classpath,
        "scala.tools.nsc.Main",
        "-d",
        str(TARGET_CLASSES_DIR),
        "-classpath",
        classpath,
    ] + [str(f) for f in scala_files]

    print("Executing Scala compilation via scala.tools.nsc.Main...")
    res = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Compilation FAILED:")
        print(res.stderr or res.stdout)
        return False
    else:
        print("Compilation succeeded!")

    # Package into jar
    package_cmd = [
        jar_bin,
        "cvf",
        str(OUTPUT_JAR),
        "-C",
        str(TARGET_CLASSES_DIR),
        "."
    ]
    print(f"Packaging compiled classes into {OUTPUT_JAR}...")
    res_pkg = subprocess.run(package_cmd, capture_output=True, text=True)
    if res_pkg.returncode != 0:
        print("Packaging FAILED:")
        print(res_pkg.stderr)
        return False

    print(f"Successfully generated: {OUTPUT_JAR}")
    return True


if __name__ == "__main__":
    success = compile_scala()
    if not success:
        sys.exit(1)
