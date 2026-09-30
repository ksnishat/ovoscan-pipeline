"""
OvoScan Logging Configuration
Structured JSON logging with request context
"""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional
from pythonjsonlogger import jsonlogger

from src.config import settings


# Context variable for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


class RequestContextFilter(logging.Filter):
    """Add request context to log records"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "N/A"
        record.user_id = user_id_var.get() or "N/A"
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""

    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)

        # Standard fields
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = settings.app_name
        log_record["environment"] = settings.app_env

        # Request context
        log_record["request_id"] = getattr(record, "request_id", "N/A")
        log_record["user_id"] = getattr(record, "user_id", "N/A")

        # Extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "message", "exc_info", "exc_text",
                "stack_info", "request_id", "user_id"
            ]:
                log_record[key] = value


def get_json_handler() -> logging.Handler:
    """Get JSON handler for structured logging"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomJsonFormatter())
    handler.addFilter(RequestContextFilter())
    return handler


def get_console_handler() -> logging.Handler:
    """Get human-readable console handler for development"""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())
    return handler


def setup_logging(
    log_level: Optional[str] = None,
    json_format: bool = True,
    include_console: bool = False
) -> logging.Logger:
    """
    Configure application logging.
    """
    level = getattr(logging, (log_level or settings.app_log_level).upper())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    if json_format or settings.app_env == "production":
        root_logger.addHandler(get_json_handler())

    if include_console or settings.app_env == "development":
        root_logger.addHandler(get_console_handler())

    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("zenml").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "log_level": level,
            "json_format": json_format,
            "environment": settings.app_env,
        }
    )

    return root_logger


def set_request_context(request_id: str = None, user_id: str = None) -> None:
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context() -> None:
    request_id_var.set(None)
    user_id_var.set(None)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_detection(
    logger: logging.Logger,
    image_id: str,
    num_detections: int,
    classes: list,
    confidences: list,
    processing_time_ms: float,
) -> None:
    """Log defect detection with structured data"""
    logger.info(
        "Defect detection completed",
        extra={
            "event_type": "defect_detection",
            "image_id": image_id,
            "num_detections": num_detections,
            "classes": classes,
            "confidences": confidences,
            "processing_time_ms": processing_time_ms,
        }
    )


def log_quality_report(
    logger: logging.Logger,
    image_id: str,
    overall_quality: str,
    defect_summary: dict,
    processing_time_ms: float,
    model: str,
) -> None:
    """Log quality assessment report"""
    logger.info(
        "Quality report generated",
        extra={
            "event_type": "quality_report",
            "image_id": image_id,
            "overall_quality": overall_quality,
            "defect_summary": defect_summary,
            "processing_time_ms": processing_time_ms,
            "model": model,
        }
    )


def log_pipeline_run(
    logger: logging.Logger,
    pipeline_name: str,
    run_id: str,
    status: str,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log ZenML pipeline run"""
    extra = {
        "event_type": "pipeline_run",
        "pipeline_name": pipeline_name,
        "run_id": run_id,
        "status": status,
    }
    if duration_ms:
        extra["duration_ms"] = duration_ms
    if error:
        extra["error"] = error
        logger.error(f"Pipeline run failed: {pipeline_name}", extra=extra)
    else:
        logger.info(f"Pipeline run completed: {pipeline_name}", extra=extra)


def log_model_operation(
    logger: logging.Logger,
    operation: str,
    model_name: str,
    version: Optional[str] = None,
    status: str = "success",
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Log model operations"""
    extra = {
        "event_type": "model_operation",
        "operation": operation,
        "model_name": model_name,
        "status": status,
    }
    if version:
        extra["version"] = version
    if duration_ms:
        extra["duration_ms"] = duration_ms
    if error:
        extra["error"] = error
        logger.error(f"Model operation failed: {operation}", extra=extra)
    else:
        logger.info(f"Model operation completed: {operation}", extra=extra)


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_size: Optional[int] = None,
    response_size: Optional[int] = None,
) -> None:
    """Log API request"""
    logger.info(
        "API request",
        extra={
            "event_type": "api_request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "request_size_bytes": request_size,
            "response_size_bytes": response_size,
        }
    )