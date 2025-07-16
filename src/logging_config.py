"""
Logging configuration and utilities for ToxTrac Data Analyzer.

This module provides centralized logging configuration using Rich for 
beautiful console output and comprehensive file logging.

Author: Aman Rathore
Contact: amanr.me | amanrathore9753 <at> gmail <dot> com
Created on: Monday, July 14, 2025 at 10:40
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure logging for the application with Rich formatting.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for persistent logging
        console_output: Whether to enable console output
        
    Returns:
        Configured logger instance
    """
    # Install rich traceback handler for better error formatting
    install(show_locals=True)
    
    # Create console for rich output
    console = Console()
    
    # Configure root logger
    logger = logging.getLogger("toxtrac_analyzer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Add console handler with Rich formatting
    if console_output:
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Rich handler format
        console_format = "%(message)s"
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        
        # Detailed format for file logging
        file_format = (
            "%(asctime)s | %(name)s | %(levelname)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger with the specified name.
    
    Args:
        name: Name for the child logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"toxtrac_analyzer.{name}")


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            # Log function call
            args_str = ", ".join([str(arg) for arg in args])
            kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            params_str = ", ".join(filter(None, [args_str, kwargs_str]))
            
            logger.debug(f"Calling {func.__name__}({params_str})")
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(
                    f"Completed {func.__name__} in {execution_time:.3f}s"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Error in {func.__name__} after {execution_time:.3f}s: {e}"
                )
                raise
                
        return wrapper
    return decorator


# Global console instance for direct rich output
console = Console()


def print_success(message: str):
    """Print a success message with green formatting."""
    console.print(f"✅ {message}", style="bold green")


def print_warning(message: str):
    """Print a warning message with yellow formatting."""
    console.print(f"⚠️  {message}", style="bold yellow")


def print_error(message: str):
    """Print an error message with red formatting."""
    console.print(f"❌ {message}", style="bold red")


def print_info(message: str):
    """Print an info message with blue formatting."""
    console.print(f"ℹ️  {message}", style="bold blue")


def print_header(title: str):
    """Print a formatted header."""
    console.print()
    console.rule(f"[bold blue]{title}[/bold blue]")
    console.print()
