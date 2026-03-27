"""
Logging utilities for Supervisor Agent

Provides formatted logging with emojis and structured output
for better debugging and user feedback.
"""

import contextvars
import logging
import time
from typing import Optional, Dict, Any
from functools import wraps
from enum import Enum
from datetime import datetime

analysis_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("analysis_id", default="")


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.analysis_id = analysis_id_var.get("")
        return True


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SupervisorLogger:
    """
    Custom logger for the supervisor agent with formatted output
    and emoji indicators for different event types.
    """
    
    def __init__(self, name: str = "SupervisorAgent"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def thinking(self, message: str, details: Optional[str] = None):
        """Log a thinking/reasoning step"""
        self.logger.info(f"🧠 THINKING | {message}")
        if details:
            self.logger.debug(f"   └─ {details}")
    
    def routing(self, agent_name: str, confidence: float, reasoning: str):
        """Log a routing decision"""
        self.logger.info(f"🎯 ROUTING | → {agent_name} (confidence: {confidence:.0%})")
        self.logger.info(f"   └─ Reason: {reasoning[:100]}...")
    
    def validating(self, message: str, missing: Optional[list] = None):
        """Log validation step"""
        self.logger.info(f"✅ VALIDATING | {message}")
        if missing:
            self.logger.warning(f"   └─ Missing inputs: {', '.join(missing)}")
    
    def executing(self, agent_name: str, inputs: Dict[str, Any]):
        """Log agent execution start"""
        input_summary = ", ".join([f"{k}={str(v)[:30]}" for k, v in inputs.items()])
        self.logger.info(f"🚀 EXECUTING | Starting {agent_name}")
        self.logger.debug(f"   └─ Inputs: {input_summary}")
    
    def progress(self, agent_name: str, step: str, progress: float):
        """Log progress update"""
        pct = int(progress * 100)
        bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
        self.logger.info(f"⏳ PROGRESS | [{bar}] {pct}% | {step}")
    
    def completed(self, agent_name: str, duration: float, outputs: Dict[str, Any]):
        """Log agent completion"""
        output_keys = list(outputs.keys())
        self.logger.info(f"✅ COMPLETED | {agent_name} finished in {duration:.1f}s")
        self.logger.info(f"   └─ Outputs: {', '.join(output_keys)}")
    
    def error(self, message: str, exception: Optional[Exception] = None):
        """Log an error"""
        self.logger.error(f"❌ ERROR | {message}")
        if exception:
            self.logger.exception(f"   └─ Exception: {str(exception)}")
    
    def user_input_needed(self, inputs: list):
        """Log when waiting for user input"""
        self.logger.info(f"📎 WAITING | Need user input: {', '.join(inputs)}")
    
    def file_received(self, filename: str, size: int):
        """Log file receipt"""
        size_kb = size / 1024
        self.logger.info(f"📁 FILE | Received: {filename} ({size_kb:.1f} KB)")
    
    def session_event(self, event: str, session_id: str):
        """Log session events"""
        self.logger.info(f"🔄 SESSION | {event} | ID: {session_id[:8]}...")


def log_execution_time(logger: Optional[SupervisorLogger] = None):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            func_name = func.__name__
            
            if logger:
                logger.logger.debug(f"⏱️ Starting {func_name}")
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                
                if logger:
                    logger.logger.debug(f"⏱️ {func_name} completed in {duration:.2f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start
                if logger:
                    logger.error(f"{func_name} failed after {duration:.2f}s", e)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            func_name = func.__name__
            
            if logger:
                logger.logger.debug(f"⏱️ Starting {func_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                
                if logger:
                    logger.logger.debug(f"⏱️ {func_name} completed in {duration:.2f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start
                if logger:
                    logger.error(f"{func_name} failed after {duration:.2f}s", e)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Create default logger instance
import asyncio
supervisor_logger = SupervisorLogger()
