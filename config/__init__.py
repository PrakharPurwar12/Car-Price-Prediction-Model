"""CarIntel configuration package."""
from .spark_config import get_spark_session, get_project_root, SparkConfig

__all__ = ["get_spark_session", "get_project_root", "SparkConfig"]

