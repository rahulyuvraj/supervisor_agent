"""
Base classes and types for agent executors.

This module defines:
- StatusType enum for different status update types
- StatusUpdate dataclass for real-time progress updates
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class StatusType(str, Enum):
    """Types of status updates that can be yielded during agent execution"""
    THINKING = "thinking"       # Understanding/analyzing query
    ROUTING = "routing"         # Matching to appropriate agent
    VALIDATING = "validating"   # Checking requirements
    EXECUTING = "executing"     # Running the agent
    PROGRESS = "progress"       # Mid-execution progress update
    COMPLETED = "completed"     # Successful completion
    ERROR = "error"             # Execution failed
    WAITING_INPUT = "waiting_input"  # Need additional user input
    INFO = "info"               # Informational message


@dataclass
class StatusUpdate:
    """
    Real-time status update for UI display.
    
    These updates are yielded by agent executors to provide
    feedback to the user during long-running operations.
    
    Attributes:
        status_type: The type of status update
        title: Short title for the update (displayed prominently)
        message: Main message content (supports markdown)
        details: Additional details (collapsible in UI)
        progress: Progress from 0.0 to 1.0 (optional)
        agent_name: Name of the agent if applicable
        timestamp: When this update was created
        generated_files: List of files generated (for completion)
        output_dir: Output directory path (for completion)
    """
    status_type: StatusType
    title: str
    message: str
    details: Optional[str] = None
    progress: Optional[float] = None
    agent_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    generated_files: Optional[List[Dict[str, Any]]] = None
    output_dir: Optional[str] = None


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

@dataclass
class OutputValidation:
    """Result of post-execution output directory scan."""
    total_files: int = 0
    total_size: int = 0
    file_types: Dict[str, int] = field(default_factory=dict)
    empty_dirs: List[str] = field(default_factory=list)
    summary: str = ""


def _format_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def validate_pipeline_output(output_dir: str, agent_display_name: str) -> OutputValidation:
    """Scan an output directory and return a validation summary."""
    from pathlib import Path

    root = Path(output_dir)
    result = OutputValidation()

    if not root.exists():
        result.summary = f"⚠️ {agent_display_name}: output directory does not exist ({output_dir})"
        return result

    for f in root.rglob("*"):
        if f.is_file():
            result.total_files += 1
            result.total_size += f.stat().st_size
            ext = f.suffix.lower() or "(no ext)"
            result.file_types[ext] = result.file_types.get(ext, 0) + 1

    for d in root.rglob("*"):
        if d.is_dir() and not any(d.iterdir()):
            result.empty_dirs.append(str(d.relative_to(root)))

    # Build summary
    lines = [f"📊 **{agent_display_name} Output Validation**"]
    lines.append(f"  Files: {result.total_files} | Size: {_format_size(result.total_size)}")
    if result.file_types:
        top = sorted(result.file_types.items(), key=lambda x: -x[1])[:6]
        lines.append("  Types: " + ", ".join(f"{ext} ({n})" for ext, n in top))
    if result.empty_dirs:
        lines.append(f"  ⚠️ Empty subdirs: {', '.join(result.empty_dirs)}")

    result.summary = "\n".join(lines)
    return result
