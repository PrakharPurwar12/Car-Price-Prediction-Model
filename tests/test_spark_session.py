"""
tests/test_spark_session.py
Validates SparkSession instantiation, Java 17 configuration,
and WinLocalFileSystem Windows Hadoop bypass functionality.
"""

import pytest
import config.spark_config
from config.spark_config import get_spark_session


@pytest.fixture(scope="module")
def spark():
    session = get_spark_session("CarIntel-TestSparkSession")
    yield session
    session.stop()


def test_spark_session_active(spark):
    assert spark is not None
    assert spark.version.startswith("4.") or spark.version.startswith("3.")


def test_win_local_fs_configured(spark):
    fs_impl = spark.conf.get("spark.hadoop.fs.file.impl", "")
    assert "com.carintel.hadoop.WinLocalFileSystem" in fs_impl


def test_basic_dataframe_operations(spark):
    data = [("Maruti", 2018, 450000.0), ("Hyundai", 2019, 550000.0)]
    columns = ["brand", "year", "selling_price"]
    df = spark.createDataFrame(data, columns)
    assert df.count() == 2
    row = df.filter(df.brand == "Maruti").first()
    assert row["year"] == 2018

