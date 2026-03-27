"""
Streamlit Chat UI for Bioinformatics Supervisor Agent

A conversational interface that:
1. Accepts user queries about bioinformatics analyses
2. Allows file uploads
3. Shows real-time progress and reasoning
4. Displays agent outputs
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

# Add project root to path FIRST (before any local imports)
# Use resolve() to get absolute path
project_root = Path(__file__).resolve().parent.parent.parent  # streamlit_app -> agentic_ai_wf -> agenticaib
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx
import platform

# Configure logging BEFORE using logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Now import from agentic_ai_wf (after path is set)
from agentic_ai_wf.supervisor_agent import (
    SupervisorAgent, SessionManager, ConversationState
)
from agentic_ai_wf.supervisor_agent.supervisor import StatusType, StatusUpdate
from agentic_ai_wf.supervisor_agent.agent_registry import AGENT_REGISTRY

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🧬 BiRAGAS ",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Main chat container */
    .main-chat {
        max-width: 900px;
        margin: 0 auto;
    }
    
    /* Message styles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background: #f0f2f6;
        color: #1f2937;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 85%;
    }
    
    /* Status update styles */
    .status-thinking {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .status-routing {
        background: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .status-executing {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .status-error {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    .status-waiting {
        background: #ede9fe;
        border-left: 4px solid #8b5cf6;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Progress bar */
    .progress-container {
        background: #e5e7eb;
        border-radius: 10px;
        height: 8px;
        margin-top: 8px;
        overflow: hidden;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #10b981, #3b82f6);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Agent card */
    .agent-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styles */
    .sidebar-section {
        background: #f8fafc;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        gap: 4px;
        padding: 8px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #9ca3af;
        border-radius: 50%;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-4px); }
    }
    
    /* Multi-agent pipeline styles */
    .pipeline-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 12px 0;
        color: white;
    }
    
    .pipeline-step {
        display: inline-flex;
        align-items: center;
        background: rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 8px 16px;
        margin: 4px;
        font-size: 14px;
    }
    
    .pipeline-step.active {
        background: #10b981;
        animation: pulse 2s infinite;
    }
    
    .pipeline-step.completed {
        background: #059669;
    }
    
    .pipeline-step.pending {
        opacity: 0.6;
    }
    
    .pipeline-arrow {
        color: #6366f1;
        font-size: 20px;
        margin: 0 8px;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        50% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    }
    
    /* Terminal-like log display */
    .terminal-log {
        background: #0f172a;
        color: #10b981;
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 13px;
        padding: 16px;
        border-radius: 8px;
        max-height: 400px;
        overflow-y: auto;
        margin: 12px 0;
    }
    
    .terminal-log .timestamp {
        color: #64748b;
    }
    
    .terminal-log .agent-name {
        color: #f59e0b;
    }
    
    .terminal-log .status {
        color: #06b6d4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize Streamlit session state"""
    if "session_manager" not in st.session_state:
        st.session_state.session_manager = SessionManager()
    
    if "supervisor" not in st.session_state:
        upload_dir = Path("./streamlit_uploads")
        upload_dir.mkdir(exist_ok=True)
        st.session_state.supervisor = SupervisorAgent(
            session_manager=st.session_state.session_manager,
            upload_dir=str(upload_dir)
        )
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = st.session_state.session_manager.create_session()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "current_status" not in st.session_state:
        st.session_state.current_status = None
    
    if "uploaded_files_cache" not in st.session_state:
        st.session_state.uploaded_files_cache = {}
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Cancellation flag for stopping running tasks
    if "cancel_requested" not in st.session_state:
        st.session_state.cancel_requested = False
    
    # Track the current async task for cancellation
    if "current_task" not in st.session_state:
        st.session_state.current_task = None
    
    # Activity log for better process visibility
    if "activity_log" not in st.session_state:
        st.session_state.activity_log = []
    
    # Track all generated files by agent
    if "generated_files" not in st.session_state:
        st.session_state.generated_files = {}  # {agent_name: {type: [files]}}

init_session_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def render_download_buttons():
    """Render download buttons for any available result files"""
    if "conversation" not in st.session_state:
        return
    
    conv = st.session_state.conversation
    if not conv.workflow_state:
        return
    
    downloadable_files = {
        "prioritized_genes_path": "Prioritized Genes",
        "pathway_consolidation_path": "Pathway Results",
    }
    
    has_files = False
    for state_key, display_name in downloadable_files.items():
        if state_key in conv.workflow_state and conv.workflow_state[state_key]:
            file_path = Path(conv.workflow_state[state_key])
            if file_path.exists() and file_path.is_file():
                has_files = True
                break
    
    if has_files:
        st.markdown("**📥 Download Results:**")
        cols = st.columns(len(downloadable_files))
        col_idx = 0
        
        for state_key, display_name in downloadable_files.items():
            if state_key in conv.workflow_state and conv.workflow_state[state_key]:
                file_path = Path(conv.workflow_state[state_key])
                if file_path.exists() and file_path.is_file():
                    try:
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        with cols[col_idx % len(cols)]:
                            st.download_button(
                                label=f"📄 {display_name}",
                                data=file_data,
                                file_name=file_path.name,
                                mime="text/csv",
                                key=f"chat_download_{state_key}_{file_path.name}"
                            )
                        col_idx += 1
                    except Exception as e:
                        pass


def _status_value(status_type: object) -> str:
    """Normalize status values across different supervisor update shapes."""
    return getattr(status_type, "value", status_type)


def _status_is(status_type: object, *expected: object) -> bool:
    current = _status_value(status_type)
    return current in {_status_value(item) for item in expected}

def get_status_class(status_type: StatusType) -> str:
    """Get CSS class for status type"""
    mapping = {
        "thinking": "status-thinking",
        "routing": "status-routing",
        "validating": "status-routing",
        "executing": "status-executing",
        "progress": "status-executing",
        "info": "status-routing",
        "completed": "status-executing",
        "error": "status-error",
        "waiting_input": "status-waiting"
    }
    return mapping.get(_status_value(status_type), "status-thinking")

def get_status_icon(status_type: StatusType) -> str:
    """Get icon for status type"""
    mapping = {
        "thinking": "🧠",
        "routing": "🔍",
        "validating": "📋",
        "executing": "⚡",
        "progress": "🔄",
        "info": "ℹ️",
        "completed": "✅",
        "error": "❌",
        "waiting_input": "📎"
    }
    return mapping.get(_status_value(status_type), "💭")

def render_status_update(update: StatusUpdate):
    """Render a status update in the UI using Streamlit native components"""
    icon = get_status_icon(update.status_type)
    
    # Map status type to Streamlit colors
    status_colors = {
        "thinking": "🟡",
        "routing": "🔵",
        "validating": "🔵",
        "executing": "🟢",
        "progress": "🟢",
        "info": "🔵",
        "completed": "🟢",
        "error": "🔴",
        "waiting_input": "🟣"
    }
    
    color_dot = status_colors.get(_status_value(update.status_type), "⚪")
    
    # Check if this is a multi-agent pipeline update
    is_pipeline = "Pipeline" in update.title or "[" in update.title
    
    # Use Streamlit's native markdown for clean rendering
    if is_pipeline:
        # Special rendering for pipeline updates
        st.markdown(f"### 🔗 {update.title}")
    else:
        st.markdown(f"**{icon} {update.title}**")
    st.markdown(update.message)
    
    # Show details only if meaningful
    if update.details and len(update.details.strip()) > 0:
        # Skip if it contains HTML or unhelpful text
        if "<div" not in update.details and "How to provide inputs" not in update.details:
            with st.expander("Details", expanded=False):
                st.markdown(update.details)
    
    # Progress bar
    if update.progress is not None:
        st.progress(update.progress)
    
    # For completed status, show generated files and plots
    if _status_is(update.status_type, StatusType.COMPLETED):
        # Display generated plots if available
        if hasattr(update, 'generated_files') and update.generated_files:
            render_generated_plots(update.generated_files)
            # Register files for sidebar downloads
            if hasattr(update, 'agent_name') and update.agent_name:
                register_generated_files(update.agent_name, update.generated_files)
        
        render_download_buttons()
    
    st.markdown("---")

def render_message(role: str, content: str):
    """Render a chat message"""
    if role == "user":
        st.markdown(f"""
        <div class="user-message">
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            {content}
        </div>
        """, unsafe_allow_html=True)

def save_uploaded_file(uploaded_file) -> str:
    """Save an uploaded file and return the path"""
    upload_dir = Path("./streamlit_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / uploaded_file.name
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return str(file_path)


# ============================================================================
# ACTIVITY LOG & FILE TRACKING HELPERS
# ============================================================================

def add_activity_log(update: StatusUpdate):
    """Add a status update to the activity log with human-readable formatting"""
    from datetime import datetime
    
    timestamp = datetime.fromtimestamp(update.timestamp).strftime("%H:%M:%S")
    agent = update.agent_name or "System"
    status = _status_value(update.status_type)
    
    # Create meaningful log entry based on status type
    log_entry = {
        "timestamp": timestamp,
        "agent": agent,
        "status": status,
        "title": update.title,
        "message": update.message[:200] + "..." if len(update.message) > 200 else update.message,
        "progress": update.progress
    }
    
    st.session_state.activity_log.append(log_entry)
    # Keep last 50 entries
    if len(st.session_state.activity_log) > 50:
        st.session_state.activity_log = st.session_state.activity_log[-50:]


def categorize_files(file_paths: list) -> dict:
    """Categorize files by type for organized display"""
    categories = {
        "plots": [],      # PNG, JPG images
        "data": [],       # CSV, TSV files
        "reports": [],    # PDF, DOCX, HTML
        "other": []
    }
    
    for fp in file_paths:
        p = Path(fp)
        ext = p.suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.svg']:
            categories["plots"].append(fp)
        elif ext in ['.csv', '.tsv', '.xlsx', '.json']:
            categories["data"].append(fp)
        elif ext in ['.pdf', '.docx', '.html']:
            categories["reports"].append(fp)
        else:
            categories["other"].append(fp)
    
    return categories


def create_zip_from_folder(folder_path: Path) -> Optional[bytes]:
    """Create a ZIP file from a folder and return bytes for download"""
    import io
    import zipfile
    
    if not folder_path.exists():
        return None
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(folder_path)
                zf.write(file_path, arcname)
    
    buffer.seek(0)
    return buffer.getvalue()


def register_generated_files(agent_name: str, file_paths: list):
    """Register generated files from an agent for tracking"""
    if not file_paths:
        return
    
    categorized = categorize_files(file_paths)
    
    if agent_name not in st.session_state.generated_files:
        st.session_state.generated_files[agent_name] = {"plots": [], "data": [], "reports": [], "other": []}
    
    for cat, files in categorized.items():
        # Avoid duplicates
        for f in files:
            if f not in st.session_state.generated_files[agent_name][cat]:
                st.session_state.generated_files[agent_name][cat].append(f)


def render_generated_plots(file_paths: list, max_display: int = 6):
    """Render generated plot images in the UI"""
    plots = [f for f in file_paths if Path(f).suffix.lower() in ['.png', '.jpg', '.jpeg']]
    
    if not plots:
        return
    
    with st.expander(f"📊 Generated Visualizations ({len(plots)} plots)", expanded=True):
        # Display in columns
        cols = st.columns(min(3, len(plots)))
        for idx, plot_path in enumerate(plots[:max_display]):
            p = Path(plot_path)
            if p.exists():
                with cols[idx % 3]:
                    try:
                        st.image(str(p), caption=p.stem, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not load: {p.name}")
        
        if len(plots) > max_display:
            st.info(f"📁 {len(plots) - max_display} more plots available in output directory")


def render_activity_log():
    """Render the activity log in a terminal-style display"""
    if not st.session_state.activity_log:
        return
    
    with st.expander("📋 Activity Log", expanded=False):
        log_html = '<div class="terminal-log">'
        for entry in reversed(st.session_state.activity_log[-20:]):
            status_emoji = {
                "thinking": "🧠", "routing": "🔍", "validating": "📋",
                "executing": "⚡", "progress": "🔄", "info": "ℹ️",
                "completed": "✅", "error": "❌", "waiting_input": "📎"
            }.get(entry["status"], "💭")
            
            progress_str = f" [{int(entry['progress']*100)}%]" if entry["progress"] else ""
            log_html += (
                f'<div><span class="timestamp">[{entry["timestamp"]}]</span> '
                f'<span class="agent-name">{entry["agent"]}</span> '
                f'{status_emoji} <span class="status">{entry["title"]}</span>{progress_str}</div>'
            )
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🧬 Bioinformatics Assistant")
    st.markdown("---")
    
    # Available Agents Section
    st.markdown("### 🤖 Available Agents")
    
    for agent_type, agent_info in AGENT_REGISTRY.items():
        with st.expander(agent_info.display_name, expanded=False):
            st.markdown(f"**{agent_info.description}**")
            st.markdown(f"⏱️ Estimated time: {agent_info.estimated_time}")
            
            st.markdown("**Required inputs:**")
            for inp in agent_info.required_inputs:
                emoji = "📎" if inp.is_file else "✏️"
                st.markdown(f"- {emoji} {inp.name}")
            
            st.markdown("**Example queries:**")
            for q in agent_info.example_queries[:2]:
                st.markdown(f"- _{q}_")
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("### 📁 Upload Files")
    
    uploaded_files = st.file_uploader(
        "Drop your data files here",
        type=["csv", "tsv", "xlsx", "h5ad", "txt", "json", "fastq", "fq", "gz", "mtx"],
        accept_multiple_files=True,
        help="Upload tabular inputs, matrix files, or sequencing reads for supervisor-driven analysis"
    )
    
    if uploaded_files:
        st.markdown("**Uploaded files:**")
        for uf in uploaded_files:
            if uf.name not in st.session_state.uploaded_files_cache:
                file_path = save_uploaded_file(uf)
                st.session_state.uploaded_files_cache[uf.name] = file_path
                logger.info(f"Saved uploaded file: {uf.name} -> {file_path}")
            
            st.markdown(f"✅ {uf.name}")
    
    st.markdown("---")
    
    # Session Info
    st.markdown("### 📊 Session Info")
    
    conv = st.session_state.conversation
    st.markdown(f"**Session ID:** `{conv.session_id[:8]}...`")
    st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    if conv.current_disease:
        st.markdown(f"**Current disease:** {conv.current_disease}")
    
    # Download Results Section
    if conv.workflow_state or st.session_state.generated_files:
        st.markdown("---")
        st.markdown("### 📥 Download Results")
        
        # Dynamic file downloads from generated_files tracking
        if st.session_state.generated_files:
            for agent_name, categories in st.session_state.generated_files.items():
                agent_display = agent_name.replace("_", " ").title()
                
                # Count total files for this agent
                total_files = sum(len(files) for files in categories.values())
                if total_files == 0:
                    continue
                
                with st.expander(f"📁 {agent_display} ({total_files} files)", expanded=False):
                    # Data files (CSV, TSV)
                    if categories.get("data"):
                        st.markdown("**📊 Data Files:**")
                        for fp in categories["data"][:10]:  # Limit display
                            p = Path(fp)
                            if p.exists():
                                try:
                                    with open(p, 'rb') as f:
                                        st.download_button(
                                            label=f"📄 {p.name}",
                                            data=f.read(),
                                            file_name=p.name,
                                            mime="text/csv" if p.suffix == '.csv' else "text/tab-separated-values",
                                            key=f"dl_{agent_name}_{p.name}",
                                            use_container_width=True
                                        )
                                except:
                                    pass
                    
                    # Plot files (PNG, JPG)
                    if categories.get("plots"):
                        st.markdown("**📈 Plots:**")
                        for fp in categories["plots"][:10]:
                            p = Path(fp)
                            if p.exists():
                                try:
                                    with open(p, 'rb') as f:
                                        st.download_button(
                                            label=f"🖼️ {p.name}",
                                            data=f.read(),
                                            file_name=p.name,
                                            mime=f"image/{p.suffix[1:]}",
                                            key=f"dl_{agent_name}_{p.name}",
                                            use_container_width=True
                                        )
                                except:
                                    pass
                    
                    # Report files (PDF, DOCX)
                    if categories.get("reports"):
                        st.markdown("**📑 Reports:**")
                        for fp in categories["reports"][:5]:
                            p = Path(fp)
                            if p.exists():
                                try:
                                    with open(p, 'rb') as f:
                                        st.download_button(
                                            label=f"📄 {p.name}",
                                            data=f.read(),
                                            file_name=p.name,
                                            mime="application/octet-stream",
                                            key=f"dl_{agent_name}_{p.name}",
                                            use_container_width=True
                                        )
                                except:
                                    pass
        
        # Fallback to workflow state based downloads
        elif conv.workflow_state:
            downloadable_files = {
                "prioritized_genes_path": ("Prioritized Genes", "csv"),
                "pathway_consolidation_path": ("Pathway Analysis", "csv"),
                "deg_base_dir": ("DEG Results", "folder"),
                "deconvolution_output_dir": ("Deconvolution Results", "folder"),
                "cohort_output_dir": ("Cohort Data", "folder"),
                "mdp_output_dir": ("MDP Analysis Results", "folder"),
                "temporal_output_dir": ("Temporal Analysis Results", "folder"),
                "harmonization_output_dir": ("Harmonization Results", "folder"),
                "reporting_output_dir": ("Report Results", "folder"),
            }
            
            for state_key, (display_name, file_type) in downloadable_files.items():
                if state_key in conv.workflow_state and conv.workflow_state[state_key]:
                    file_path = conv.workflow_state[state_key]
                    
                    if file_type == "csv" and Path(file_path).exists() and Path(file_path).is_file():
                        try:
                            with open(file_path, 'rb') as f:
                                file_data = f.read()
                            st.download_button(
                                label=f"📄 {display_name}",
                                data=file_data,
                                file_name=Path(file_path).name,
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"Could not load {display_name}")
                    elif file_type == "folder" and Path(file_path).exists():
                        folder_path = Path(file_path)
                        # Create ZIP download button
                        zip_data = create_zip_from_folder(folder_path)
                        if zip_data:
                            st.download_button(
                                label=f"📦 {display_name} (ZIP)",
                                data=zip_data,
                                file_name=f"{folder_path.name}.zip",
                                mime="application/zip",
                                key=f"zip_{state_key}",
                                use_container_width=True
                            )
                        # Also show expandable list of key files
                        with st.expander(f"📁 {display_name} (browse files)"):
                            all_files = list(folder_path.glob("**/*.csv")) + list(folder_path.glob("**/*.pdf")) + list(folder_path.glob("**/*.html"))
                            for f in all_files[:10]:
                                try:
                                    with open(f, 'rb') as file:
                                        st.download_button(
                                            label=f"📄 {f.name}",
                                            data=file.read(),
                                            file_name=f.name,
                                            mime="application/octet-stream",
                                            key=f"dl_{state_key}_{f.name}"
                                        )
                                except:
                                    pass
    
    # Structured report notification
    if conv.workflow_state and conv.workflow_state.get("structured_report_md_path"):
        st.markdown("---")
        st.markdown("### 📊 Structured Report")
        report_path = Path(conv.workflow_state["structured_report_md_path"])
        if report_path.exists():
            st.success(f"Report generated: **{report_path.name}**")
            st.page_link("pages/1_structured_report.py", label="📊 Open Report Viewer", icon="📊")
        else:
            st.info("Report file not found on disk.")
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation = st.session_state.session_manager.create_session()
        st.session_state.uploaded_files_cache = {}
        st.session_state.activity_log = []
        st.session_state.generated_files = {}
        st.rerun()

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

st.markdown("# 🧬 BiRAGAS")
st.markdown("*Your intelligent guide to transcriptome analysis*")
st.markdown("---")

# Display chat history
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        elif msg["role"] == "status":
            render_status_update(msg["update"])

# Show activity log if there are entries
if st.session_state.activity_log:
    render_activity_log()

# Status placeholder for live updates
status_placeholder = st.empty()

# Stop button when processing
if st.session_state.processing:
    stop_col1, stop_col2 = st.columns([3, 1])
    with stop_col2:
        if st.button("🛑 Stop Processing", type="secondary", use_container_width=True):
            st.session_state.cancel_requested = True
            st.warning("⏳ Cancellation requested... waiting for current step to complete.")

# Chat input
if prompt := st.chat_input("Ask me about bioinformatics analysis...", disabled=st.session_state.processing):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Process the message
    st.session_state.processing = True
    st.session_state.cancel_requested = False  # Reset cancellation flag
    
    # Prepare uploaded files
    uploaded_files_dict = st.session_state.uploaded_files_cache.copy()
    
    async def process_message():
        """Process the user message asynchronously"""
        supervisor = st.session_state.supervisor
        final_message = None
        all_status_updates = []
        current_agent = None
        agent_logs = []  # Track logs per agent
        pipeline_info = None  # Store the pipeline plan when we see it
        system_logs = []  # For routing/planning messages (no agent_name)
        was_cancelled = False
        
        # TODO: set st.session_state.user_id from auth middleware
        async for update in supervisor.process_message(
            user_message=prompt,
            session_id=st.session_state.conversation.session_id,
            uploaded_files=uploaded_files_dict,
            user_id=st.session_state.get("user_id"),
        ):
            # Check for cancellation request at each update
            if st.session_state.cancel_requested:
                was_cancelled = True
                logger.info("Processing cancelled by user")
                with status_placeholder.container():
                    st.warning("🛑 **Processing Cancelled**\n\nThe operation was stopped by user request.")
                break
            
            all_status_updates.append(update)
            
            # Add to activity log for visibility
            add_activity_log(update)
            
            # Capture pipeline plan from ROUTING status updates
            if _status_is(update.status_type, StatusType.ROUTING) and "agents" in update.message.lower():
                pipeline_info = update.message
            
            # Handle status updates with agent_name vs system logs
            if update.agent_name:
                # Track agent changes for structured display
                if update.agent_name != current_agent:
                    current_agent = update.agent_name
                    agent_logs.append({"agent": current_agent, "steps": [], "completed": False})
                
                # Add step to current agent's log
                if agent_logs:
                    agent_logs[-1]["steps"].append({
                        "icon": get_status_icon(update.status_type),
                        "title": update.title,
                        "message": update.message,
                        "progress": update.progress,
                        "status": _status_value(update.status_type)
                    })
                    
                    # Mark as completed if this is a completion status
                    if _status_is(update.status_type, StatusType.COMPLETED):
                        agent_logs[-1]["completed"] = True
            else:
                # System log (routing, planning, etc.)
                system_logs.append({
                    "icon": get_status_icon(update.status_type),
                    "title": update.title,
                    "message": update.message
                })
            
            # Show real-time structured log in the placeholder
            with status_placeholder.container():
                # Header showing what's happening
                st.markdown("### 🔄 Processing Your Request")
                st.markdown("---")
                
                # Show pipeline plan if available
                if pipeline_info:
                    st.info(f"📋 **Pipeline Plan**\n\n{pipeline_info}")
                    st.markdown("")
                
                # Show system logs (routing/planning) if no agents yet
                if not agent_logs and system_logs:
                    for log in system_logs[-3:]:  # Show last 3 system logs
                        st.markdown(f"{log['icon']} **{log['title']}**")
                        if log['message']:
                            st.markdown(f"  _{log['message'][:200]}_")
                    st.markdown("")
                
                # Show each agent and its steps
                for idx, agent_log in enumerate(agent_logs):
                    agent_name = agent_log["agent"].replace("_", " ").title()
                    is_current = (idx == len(agent_logs) - 1) and not agent_log["completed"]
                    is_completed = agent_log["completed"]
                    
                    # Agent header with status indicator
                    if is_completed:
                        st.markdown(f"#### ✅ **{agent_name}**")
                    elif is_current:
                        st.markdown(f"#### ⚡ **{agent_name}** _(running...)_")
                    else:
                        st.markdown(f"#### 🔹 **{agent_name}**")
                    
                    # Show steps for this agent (last 5 for current, last 2 for completed)
                    if is_current:
                        steps_to_show = agent_log["steps"][-5:]
                    elif is_completed:
                        steps_to_show = agent_log["steps"][-2:]
                    else:
                        steps_to_show = agent_log["steps"][-1:]
                    
                    for step in steps_to_show:
                        # Format step with icon and message
                        step_icon = step["icon"]
                        step_msg = step["message"]
                        # Truncate long messages
                        if len(step_msg) > 150:
                            step_msg = step_msg[:147] + "..."
                        
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{step_icon} {step_msg}")
                    
                    # Show progress bar for current agent
                    if is_current and update.progress is not None:
                        st.progress(update.progress)
                    
                    st.markdown("")  # Spacer
                
                # Show latest details in expander
                if update.details and len(update.details.strip()) > 0:
                    if "<div" not in update.details and "How to provide inputs" not in update.details:
                        with st.expander("📋 Current Step Details", expanded=False):
                            st.markdown(update.details)
                
                # Show generated plots inline during processing
                if hasattr(update, 'generated_files') and update.generated_files:
                    plots = [f for f in update.generated_files if Path(f).suffix.lower() in ['.png', '.jpg', '.jpeg']]
                    if plots:
                        st.markdown("**📊 Generated Visualizations:**")
                        cols = st.columns(min(3, len(plots)))
                        for idx, plot_path in enumerate(plots[:6]):
                            p = Path(plot_path)
                            if p.exists():
                                with cols[idx % 3]:
                                    try:
                                        st.image(str(p), caption=p.stem, use_container_width=True)
                                    except:
                                        pass
            
            # Store status update for history (skip COMPLETED — it becomes the assistant message below)
            if not _status_is(update.status_type, StatusType.COMPLETED):
                st.session_state.messages.append({
                    "role": "status",
                    "update": update
                })
            
            # Track final message and register files on completion
            if _status_is(update.status_type, StatusType.COMPLETED):
                final_message = update.message
                agent_name = update.agent_name or "unknown_agent"
                
                # Register generated files for download tracking
                if hasattr(update, 'generated_files') and update.generated_files:
                    register_generated_files(agent_name, update.generated_files)
            elif _status_is(update.status_type, StatusType.ERROR, StatusType.WAITING_INPUT):
                final_message = update.message
        
        # Clear the live status placeholder after completion
        status_placeholder.empty()
        
        # Handle cancellation message
        if was_cancelled:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "🛑 Processing was cancelled. You can start a new request when ready."
            })
        elif final_message:
            # Add final assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_message
            })
        
        st.session_state.processing = False
        st.session_state.cancel_requested = False  # Reset for next run
    
    # Run async processing
    asyncio.run(process_message())
    
    # Rerun to update UI
    st.rerun()

# ============================================================================
# WELCOME MESSAGE
# ============================================================================

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 40px;">
        <h2>👋 Welcome!</h2>
        <p style="font-size: 1.1em; color: #6b7280;">
            I'm your AI-powered bioinformatics assistant. I can help you with:
        </p>
        <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px;">
            <div style="background: #f0fdf4; padding: 15px; border-radius: 10px; width: 200px;">
                <div style="font-size: 2em;">🔍</div>
                <div style="font-weight: 600;">Cohort Retrieval</div>
                <div style="font-size: 0.85em; color: #6b7280;">Search public databases</div>
            </div>
            <div style="background: #eff6ff; padding: 15px; border-radius: 10px; width: 200px;">
                <div style="font-size: 2em;">📊</div>
                <div style="font-weight: 600;">DEG Analysis</div>
                <div style="font-size: 0.85em; color: #6b7280;">Differential expression</div>
            </div>
            <div style="background: #fef3c7; padding: 15px; border-radius: 10px; width: 200px;">
                <div style="font-size: 2em;">🎯</div>
                <div style="font-weight: 600;">Gene Prioritization</div>
                <div style="font-size: 0.85em; color: #6b7280;">Rank by relevance</div>
            </div>
            <div style="background: #ede9fe; padding: 15px; border-radius: 10px; width: 200px;">
                <div style="font-size: 2em;">🛤️</div>
                <div style="font-weight: 600;">Pathway Analysis</div>
                <div style="font-size: 0.85em; color: #6b7280;">Enrichment analysis</div>
            </div>
            <div style="background: #fce7f3; padding: 15px; border-radius: 10px; width: 200px;">
                <div style="font-size: 2em;">🧬</div>
                <div style="font-weight: 600;">Deconvolution</div>
                <div style="font-size: 0.85em; color: #6b7280;">Cell type estimation</div>
            </div>
            <div style="background: #f0f9ff; padding: 15px; border-radius: 10px; width: 200px;">
                <div style="font-size: 2em;">📊</div>
                <div style="font-weight: 600;">Structured Reports</div>
                <div style="font-size: 0.85em; color: #6b7280;">Evidence-traced reports</div>
            </div>
        </div>
        <p style="margin-top: 30px; color: #9ca3af;">
            💡 <strong>Try asking:</strong> "Find datasets for lupus disease" or "Generate a structured report for my analysis"
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.85em;'>"
    "🧬 Bioinformatics AI Assistant | Powered by LangGraph & OpenAI"
    "</div>",
    unsafe_allow_html=True
)
