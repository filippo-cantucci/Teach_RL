#!/usr/bin/env python3
# filepath: /home/filippo/0.codice/Teach_RL/logger.py

import logging
import os
import yaml
import datetime
from logging.handlers import RotatingFileHandler

# Dictionary to store loggers for different modules
loggers = {}

def setup_logging(config=None):
    """
    Setup logging configuration based on config.yaml
    If config not provided, it will be loaded from the default location
    """
    if config is None:
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            # Default configuration if config.yaml is not accessible
            config = {
                "logging": {
                    "level": "INFO",
                    "file_level": "DEBUG",
                    "console_level": "INFO",
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "file_enabled": True,
                    "max_file_size_mb": 5,
                    "backup_count": 3
                }
            }
    
    # Get logging configuration from config, or use defaults
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    file_level = getattr(logging, log_config.get("file_level", "DEBUG"))
    console_level = getattr(logging, log_config.get("console_level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_enabled = log_config.get("file_enabled", True)
    max_size_mb = log_config.get("max_file_size_mb", 5) * 1024 * 1024  # Convert to bytes
    backup_count = log_config.get("backup_count", 3)
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Create a formatter
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates on reloads
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if enabled)
    if file_enabled:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(logs_dir, f"teach_rl_{timestamp}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb,
            backupCount=backup_count
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Return the root logger
    return root_logger

def get_logger(name):
    """
    Get a logger for the specified module.
    If the logger doesn't exist, create it.
    """
    if name not in loggers:
        logger = logging.getLogger(name)
        loggers[name] = logger
    
    return loggers[name]
