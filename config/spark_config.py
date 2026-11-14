"""Unified SparkSession and Environment Configuration for CarIntel."""
import os
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the CarIntel project root."""
    return Path(__file__).resolve().parent.parent


class SparkConfig:
    """Manages paths, environment variables, and SparkSession instantiation."""

    PROJECT_ROOT = get_project_root()
    SPARK_HOME = os.environ.get("SPARK_HOME", r"C:\spark\spark-4.2.0-bin-hadoop3")
    JAVA_HOME = os.environ.get("JAVA_HOME", r"C:\Program Files\Java\jdk-17")
    WIN_FS_JAR = str(PROJECT_ROOT / "config" / "win_local_fs.jar")

    @classmethod
    def setup_environment(cls):
        """Configure system environment variables required by Spark on Windows."""
        if "JAVA_HOME" not in os.environ and os.path.exists(cls.JAVA_HOME):
            os.environ["JAVA_HOME"] = cls.JAVA_HOME
            
        if "SPARK_HOME" not in os.environ and os.path.exists(cls.SPARK_HOME):
            os.environ["SPARK_HOME"] = cls.SPARK_HOME

        current_python = sys.executable
        os.environ["PYSPARK_PYTHON"] = current_python
        os.environ["PYSPARK_DRIVER_PYTHON"] = current_python

        # Inject PySpark from SPARK_HOME if not already in sys.path
        spark_python = os.path.join(cls.SPARK_HOME, "python")
        spark_py4j_dir = os.path.join(cls.SPARK_HOME, "python", "lib")
        if os.path.exists(spark_python) and spark_python not in sys.path:
            sys.path.insert(0, spark_python)
        if os.path.exists(spark_py4j_dir):
            for item in os.listdir(spark_py4j_dir):
                if item.startswith("py4j") and item.endswith(".zip"):
                    py4j_path = os.path.join(spark_py4j_dir, item)
                    if py4j_path not in sys.path:
                        sys.path.insert(0, py4j_path)


# Automatically configure environment on import
SparkConfig.setup_environment()


def get_spark_session(app_name: str = "CarIntel-Platform", master: str = "local[*]", memory: str = "2g"):
    """
    Build or retrieve a configured SparkSession with Windows and Java 17 compatibility.
    
    Args:
        app_name: Name of the Spark application.
        master: Spark master URL (defaults to local[*]).
        memory: Driver memory limit (defaults to 2g).
        
    Returns:
        Active SparkSession instance.
    """
    SparkConfig.setup_environment()
    
    from pyspark.sql import SparkSession

    builder = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", memory) \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.default.parallelism", "4") \
        .config("spark.ui.enabled", "false") \
        .config("spark.pyspark.python", sys.executable) \
        .config("spark.pyspark.driver.python", sys.executable)

    # Attach WinLocalFileSystem to bypass winutils.exe on Windows
    jar_path = SparkConfig.WIN_FS_JAR
    if os.path.exists(jar_path):
        builder = builder.config("spark.jars", jar_path) \
                         .config("spark.driver.extraClassPath", jar_path) \
                         .config("spark.executor.extraClassPath", jar_path) \
                         .config("spark.hadoop.fs.file.impl", "com.carintel.hadoop.WinLocalFileSystem")

    # JVM options for Java 17 module encapsulation
    java_options = (
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
        "--add-opens=java.base/java.io=ALL-UNNAMED "
        "--add-opens=java.base/java.net=ALL-UNNAMED "
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
        "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
    )
    builder = builder.config("spark.driver.extraJavaOptions", java_options) \
                     .config("spark.executor.extraJavaOptions", java_options)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
