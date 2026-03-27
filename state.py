"""
Conversation State Management

Manages the state of user conversations including:
- Session tracking
- Uploaded files
- Agent outputs that can be chained
- Conversation history
"""

import logging
import os
import uuid
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

_MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))


class MessageRole(str, Enum):
    """Role of a message in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    AGENT_STATUS = "agent_status"  # For real-time updates


class MessageType(str, Enum):
    """Type of message content"""
    TEXT = "text"
    FILE_UPLOAD = "file_upload"
    AGENT_ROUTING = "agent_routing"
    AGENT_PROGRESS = "agent_progress"
    AGENT_RESULT = "agent_result"
    ERROR = "error"
    THINKING = "thinking"


@dataclass
class Message:
    """A single message in the conversation"""
    role: MessageRole
    content: str
    message_type: MessageType = MessageType.TEXT
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "type": self.message_type.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class UploadedFile:
    """Represents a file uploaded by the user"""
    filename: str
    filepath: str
    file_type: str
    upload_time: float
    size_bytes: int
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "file_type": self.file_type,
            "upload_time": self.upload_time,
            "size_bytes": self.size_bytes,
            "description": self.description
        }


@dataclass
class AgentExecution:
    """Record of an agent execution"""
    agent_name: str
    agent_display_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"  # running, completed, failed
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_display_name": self.agent_display_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": self.status,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "duration_seconds": self.duration_seconds
        }


@dataclass
class ConversationState:
    """
    Complete state of a user conversation/session.
    
    This tracks:
    - Conversation history
    - Uploaded files
    - Agent execution outputs
    - Workflow state that can be passed between agents
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # Conversation
    messages: List[Message] = field(default_factory=list)
    
    # Files uploaded by user
    uploaded_files: Dict[str, UploadedFile] = field(default_factory=dict)
    
    # Agent execution history
    agent_executions: List[AgentExecution] = field(default_factory=list)
    
    # Workflow state - outputs from agents that can be used by others
    workflow_state: Dict[str, Any] = field(default_factory=dict)
    
    # Current context
    current_disease: Optional[str] = None
    current_agent: Optional[str] = None
    waiting_for_input: bool = False
    required_inputs: List[str] = field(default_factory=list)
    pending_agent_type: Optional[str] = None  # Agent waiting for inputs
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        metadata: Dict[str, Any] = None
    ) -> Message:
        """Add a message to the conversation"""
        msg = Message(
            role=role,
            content=content,
            message_type=message_type,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.last_activity = time.time()
        return msg
    
    def add_user_message(self, content: str, metadata: Dict[str, Any] = None) -> Message:
        """Add a user message"""
        return self.add_message(MessageRole.USER, content, MessageType.TEXT, metadata)
    
    def add_assistant_message(self, content: str, metadata: Dict[str, Any] = None) -> Message:
        """Add an assistant message"""
        return self.add_message(MessageRole.ASSISTANT, content, MessageType.TEXT, metadata)
    
    def add_status_message(self, content: str, metadata: Dict[str, Any] = None) -> Message:
        """Add a status/progress message"""
        return self.add_message(MessageRole.AGENT_STATUS, content, MessageType.AGENT_PROGRESS, metadata)
    
    def add_thinking_message(self, content: str) -> Message:
        """Add a thinking/reasoning message"""
        return self.add_message(MessageRole.ASSISTANT, content, MessageType.THINKING)
    
    def add_uploaded_file(
        self,
        filename: str,
        filepath: str,
        file_type: str,
        size_bytes: int,
        description: Optional[str] = None
    ) -> UploadedFile:
        """Register an uploaded file"""
        uploaded = UploadedFile(
            filename=filename,
            filepath=filepath,
            file_type=file_type,
            upload_time=time.time(),
            size_bytes=size_bytes,
            description=description
        )
        self.uploaded_files[filename] = uploaded
        self.last_activity = time.time()
        
        logger.info(f"📁 File uploaded: {filename} ({size_bytes} bytes)")
        return uploaded
    
    def start_agent_execution(
        self,
        agent_name: str,
        agent_display_name: str,
        inputs: Dict[str, Any]
    ) -> AgentExecution:
        """Start tracking an agent execution"""
        execution = AgentExecution(
            agent_name=agent_name,
            agent_display_name=agent_display_name,
            start_time=time.time(),
            inputs=inputs
        )
        self.agent_executions.append(execution)
        self.current_agent = agent_name
        self.last_activity = time.time()
        
        logger.info(f"🚀 Started agent: {agent_display_name}")
        return execution
    
    def complete_agent_execution(
        self,
        outputs: Dict[str, Any],
        status: str = "completed"
    ):
        """Complete the current agent execution"""
        if self.agent_executions:
            execution = self.agent_executions[-1]
            execution.end_time = time.time()
            execution.status = status
            execution.outputs = outputs
            
            # Update workflow state with outputs
            self.workflow_state.update(outputs)
            self.current_agent = None
            self.last_activity = time.time()
            
            logger.info(f"✅ Completed agent: {execution.agent_display_name} in {execution.duration_seconds:.1f}s")
    
    def fail_agent_execution(self, error: str):
        """Mark current agent execution as failed"""
        if self.agent_executions:
            execution = self.agent_executions[-1]
            execution.end_time = time.time()
            execution.status = "failed"
            execution.error = error
            self.current_agent = None
            self.last_activity = time.time()
            
            logger.error(f"❌ Agent failed: {execution.agent_display_name} - {error}")
    
    def add_agent_log(self, log_message: str):
        """Add a log message to current agent execution"""
        if self.agent_executions:
            self.agent_executions[-1].logs.append(log_message)
    
    def get_available_inputs(self) -> Dict[str, Any]:
        """Get all available inputs from uploads and workflow state"""
        available = {}
        
        # From workflow state (agent outputs)
        available.update(self.workflow_state)
        
        # From uploaded files
        for filename, file_info in self.uploaded_files.items():
            filepath = file_info.filepath
            filename_lower = filename.lower()
            
            # Map common file types to input names
            if "count" in filename_lower or "expression" in filename_lower:
                available["counts_file"] = filepath
                available["bulk_file"] = filepath
            elif "meta" in filename_lower:
                available["metadata_file"] = filepath
            elif any(x in filename_lower for x in ["deg", "prioritized", "filtered"]):
                # DEG/prioritized/filtered files - map to BOTH inputs
                from pathlib import Path
                available["deg_input_file"] = filepath
                available["prioritized_genes_path"] = filepath  # Also set for pathway enrichment
                available["deg_base_dir"] = str(Path(filepath).parent)
            elif "pathway" in filename_lower:
                available["pathway_consolidation_path"] = filepath
            elif filename.endswith(".h5ad"):
                available["h5ad_file"] = filepath
            
            # Generic CSV file - use as fallback for multiple inputs
            if filename.endswith(".csv"):
                if "counts_file" not in available:
                    available["counts_file"] = filepath
                    available["bulk_file"] = filepath
                # Also use as prioritized_genes_path if not already set
                if "prioritized_genes_path" not in available:
                    available["prioritized_genes_path"] = filepath
            
            # Also store by filename for direct reference
            available[filename] = filepath
        
        # Disease name
        if self.current_disease:
            available["disease_name"] = self.current_disease
        
        return available
    
    def get_conversation_summary(self, last_n: int = 10) -> str:
        """Get a summary of recent conversation for context"""
        recent = self.messages[-last_n:] if len(self.messages) > last_n else self.messages
        
        summary_parts = []
        for msg in recent:
            if msg.message_type in [MessageType.TEXT, MessageType.AGENT_RESULT]:
                role = "User" if msg.role == MessageRole.USER else "Assistant"
                summary_parts.append(f"{role}: {msg.content[:200]}...")
        
        return "\n".join(summary_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "messages": [m.to_dict() for m in self.messages],
            "uploaded_files": {k: v.to_dict() for k, v in self.uploaded_files.items()},
            "agent_executions": [e.to_dict() for e in self.agent_executions],
            "workflow_state": self.workflow_state,
            "current_disease": self.current_disease,
            "current_agent": self.current_agent,
            "waiting_for_input": self.waiting_for_input,
            "required_inputs": self.required_inputs
        }


class SessionManager:
    """
    Manages multiple conversation sessions.
    
    In production, this would be backed by Redis or a database.
    For now, it's an in-memory store.
    """
    
    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}
        self._user_sessions: Dict[str, str] = {}  # user_id -> session_id
        logger.info("📦 SessionManager initialized")
    
    def create_session(self, user_id: Optional[str] = None) -> ConversationState:
        """Create a new session"""
        if len(self._sessions) >= _MAX_SESSIONS:
            self.cleanup_old_sessions(max_age_hours=1)
        if len(self._sessions) >= _MAX_SESSIONS:
            oldest = min(self._sessions, key=lambda sid: self._sessions[sid].last_activity)
            self.delete_session(oldest)
        session = ConversationState(user_id=user_id)
        self._sessions[session.session_id] = session
        
        if user_id:
            self._user_sessions[user_id] = session.session_id
        
        logger.info(f"🆕 Created session: {session.session_id} for user: {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationState]:
        """Get a session by ID"""
        return self._sessions.get(session_id)
    
    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> ConversationState:
        """Get existing session or create new one"""
        # Try by session_id
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if user_id and session.user_id and session.user_id != user_id:
                logger.warning("Session %s ownership mismatch — creating new session", session_id[:8])
                return self.create_session(user_id)
            return session
        
        # Try by user_id
        if user_id and user_id in self._user_sessions:
            existing_session_id = self._user_sessions[user_id]
            if existing_session_id in self._sessions:
                return self._sessions[existing_session_id]
        
        # Create new
        return self.create_session(user_id)
    
    def delete_session(self, session_id: str):
        """Delete a session"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if session.user_id and session.user_id in self._user_sessions:
                del self._user_sessions[session.user_id]
            del self._sessions[session_id]
            logger.info(f"🗑️ Deleted session: {session_id}")
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove sessions older than max_age_hours"""
        cutoff = time.time() - (max_age_hours * 3600)
        to_delete = [
            sid for sid, session in self._sessions.items()
            if session.last_activity < cutoff
        ]
        
        for sid in to_delete:
            self.delete_session(sid)
        
        if to_delete:
            logger.info(f"🧹 Cleaned up {len(to_delete)} old sessions")
