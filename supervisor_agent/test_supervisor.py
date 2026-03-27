#!/usr/bin/env python3
"""
Test script for the Supervisor Agent

Tests:
1. Intent routing
2. Agent registration
3. State management
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_routing():
    """Test the intent router"""
    from agentic_ai_wf.supervisor_agent import IntentRouter, ConversationState
    
    print("\n" + "="*60)
    print("🧪 Testing Intent Router")
    print("="*60)
    
    router = IntentRouter()
    state = ConversationState()
    
    test_queries = [
        "Find datasets for lupus disease",
        "Run DEG analysis on my count data",
        "Prioritize the genes from my DEG results",
        "What pathways are enriched in my gene list?",
        "Run cell type deconvolution using CIBERSORT",
        "Hello, what can you do?",
        "Help me analyze my data"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        decision = await router.route(query, state)
        print(f"   🎯 Agent: {decision.agent_name or 'General Query'}")
        print(f"   📊 Confidence: {decision.confidence:.0%}")
        print(f"   💭 Reasoning: {decision.reasoning[:80]}...")
        if decision.extracted_params:
            print(f"   📦 Extracted: {decision.extracted_params}")


async def test_supervisor():
    """Test the full supervisor agent"""
    from agentic_ai_wf.supervisor_agent import SupervisorAgent, SessionManager
    from agentic_ai_wf.supervisor_agent.supervisor import StatusType
    
    print("\n" + "="*60)
    print("🧪 Testing Supervisor Agent")
    print("="*60)
    
    session_manager = SessionManager()
    supervisor = SupervisorAgent(session_manager=session_manager)
    
    # Test 1: General query
    print("\n📝 Test 1: General query")
    async for update in supervisor.process_message("Hello, what can you do?"):
        print(f"   {update.status_type.value}: {update.title}")
        if update.status_type == StatusType.COMPLETED:
            print(f"   Response: {update.message[:100]}...")
    
    # Test 2: Cohort retrieval (will ask for more info)
    print("\n📝 Test 2: Cohort retrieval request")
    async for update in supervisor.process_message("Find datasets for breast cancer"):
        print(f"   {update.status_type.value}: {update.title}")
        if update.details:
            print(f"   Details: {update.details[:80]}...")
    
    # Test 3: DEG analysis without files (should request files)
    print("\n📝 Test 3: DEG analysis without files")
    async for update in supervisor.process_message("Run DEG analysis for lupus"):
        print(f"   {update.status_type.value}: {update.title}")
        if update.status_type == StatusType.WAITING_INPUT:
            print(f"   Missing inputs requested: {update.message[:100]}...")


def test_agent_registry():
    """Test agent registry"""
    from agentic_ai_wf.supervisor_agent import AGENT_REGISTRY, AgentInfo
    
    print("\n" + "="*60)
    print("🧪 Testing Agent Registry")
    print("="*60)
    
    print(f"\n📋 Registered agents: {len(AGENT_REGISTRY)}")
    
    for agent_type, agent_info in AGENT_REGISTRY.items():
        print(f"\n{agent_info.display_name}")
        print(f"   Type: {agent_type.value}")
        print(f"   Required inputs: {[i.name for i in agent_info.required_inputs]}")
        print(f"   Outputs: {[o.name for o in agent_info.outputs]}")
        print(f"   Keywords: {agent_info.keywords[:5]}...")


def test_state_management():
    """Test conversation state"""
    from agentic_ai_wf.supervisor_agent import ConversationState, SessionManager
    from agentic_ai_wf.supervisor_agent.state import MessageRole, MessageType
    
    print("\n" + "="*60)
    print("🧪 Testing State Management")
    print("="*60)
    
    state = ConversationState()
    
    # Add messages
    state.add_user_message("Find datasets for lupus")
    state.add_assistant_message("I'll search GEO for lupus datasets...")
    state.add_status_message("Searching GEO database...")
    
    print(f"\n📝 Messages: {len(state.messages)}")
    for msg in state.messages:
        print(f"   {msg.role.value}: {msg.content[:50]}...")
    
    # Add file
    state.add_uploaded_file(
        filename="counts.csv",
        filepath="/tmp/counts.csv",
        file_type="csv",
        size_bytes=1024
    )
    print(f"\n📁 Uploaded files: {list(state.uploaded_files.keys())}")
    
    # Test workflow state
    state.workflow_state["deg_base_dir"] = "/path/to/deg"
    state.current_disease = "lupus"
    
    available = state.get_available_inputs()
    print(f"\n📦 Available inputs: {list(available.keys())}")
    
    # Test session manager
    manager = SessionManager()
    session = manager.create_session("user123")
    print(f"\n🔄 Session created: {session.session_id[:8]}...")
    
    retrieved = manager.get_session(session.session_id)
    print(f"   Retrieved: {retrieved is not None}")


async def main():
    """Run all tests"""
    print("\n" + "🧬"*30)
    print("   SUPERVISOR AGENT TEST SUITE")
    print("🧬"*30)
    
    try:
        test_agent_registry()
        test_state_management()
        await test_routing()
        await test_supervisor()
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
