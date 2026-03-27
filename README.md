# 🎯 Supervisor Agent for Bioinformatics Analysis

A LangGraph-based supervisor agent that intelligently routes user queries to specialized bioinformatics analysis agents with detailed logging, reasoning explanations, and user engagement.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT CHAT UI                             │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  User: "Run pathway analysis on my DEG file"                    │ │
│  │  Bot: "🧠 Understanding... 🎯 Routing to Pathway Agent..."      │ │
│  │  Bot: "📎 I need a prioritized DEGs CSV file. Please upload."   │ │
│  │  User: [uploads file]                                           │ │
│  │  Bot: "🚀 Running pathway agent... ✅ Complete!"                │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      SUPERVISOR AGENT (LLM-based)                     │
│  • Intent Router: Understands user queries via GPT-4                  │
│  • Agent Registry: Knows all agents and their requirements            │
│  • State Manager: Tracks conversation, files, outputs                 │
│  • Provides detailed reasoning for routing decisions                  │
└───────┬────────┬────────┬──────────┬────────────┬────────────────────┘
        │        │        │          │            │
        ▼        ▼        ▼          ▼            ▼
   ┌────────┐┌────────┐┌─────────┐┌─────────┐┌──────────┐
   │Cohort  ││  DEG   ││Gene Pri-││Pathway  ││ Deconv   │
   │Retriev.││Pipeline││oritize  ││ Agent   ││ Agent    │
   └────────┘└────────┘└─────────┘└─────────┘└──────────┘
```

## 📁 Module Structure

```
supervisor_agent/
├── __init__.py              # Module exports
├── agent_registry.py        # Agent definitions and capabilities
├── state.py                 # Conversation state management
├── router.py                # LLM-based intent routing
├── supervisor.py            # Main supervisor orchestrator
├── logging_utils.py         # Formatted logging utilities
├── test_supervisor.py       # Test suite
└── README.md                # This file

streamlit_app/
├── __init__.py
└── app.py                   # Streamlit chat interface
```

## 🚀 Quick Start

### Running the Streamlit App

```bash
# From the agenticaib directory
cd /home/wrahul/projects/agenticaib

# Option 1: Using the run script
python run_streamlit.py

# Option 2: Direct streamlit command
streamlit run agentic_ai_wf/streamlit_app/app.py
```

### Testing the Supervisor

```bash
# Run test suite
python -m agentic_ai_wf.supervisor_agent.test_supervisor
```

## 🤖 Available Agents

| Agent | Description | Required Inputs | Outputs |
|-------|-------------|-----------------|---------|
| **🔍 Cohort Retrieval** | Searches GEO/ArrayExpress | `disease_name` | Dataset directory |
| **📊 DEG Analysis** | Differential expression | `counts_file`, `disease_name` | DEG results |
| **🎯 Gene Prioritization** | Ranks genes by relevance | `deg_base_dir`, `disease_name` | Prioritized genes |
| **🛤️ Pathway Enrichment** | Pathway analysis | `prioritized_genes_path`, `disease_name` | Pathway results |
| **🧬 Deconvolution** | Cell type estimation | `bulk_file`, `disease_name` | Cell proportions |

## 💬 Example Conversations

### Example 1: Dataset Search
```
User: "Find datasets for lupus disease in blood"

🧠 Understanding Your Request
   Analyzing your query to determine the best course of action...

🔍 Analyzing Intent
   Matching your request to specialized bioinformatics agents...

🎯 Routing to 🔍 Cohort Retrieval Agent
   I've identified this as a cohort retrieval task.
   
   **Why this agent?**
   The query contains keywords "find", "datasets", and specifies a disease
   (lupus) and tissue (blood), which are exactly what the cohort retrieval
   agent needs to search GEO and ArrayExpress databases.
   
   **Confidence:** 95%

📋 Checking Requirements
   ✅ disease_name: lupus (extracted from query)
   ✅ tissue_filter: blood (extracted from query)

🚀 Starting Cohort Retrieval Agent
   All requirements satisfied! Beginning analysis...

🔎 Searching GEO Database
   Querying Gene Expression Omnibus for lupus datasets...

🔎 Searching ArrayExpress
   Expanding search to ArrayExpress database...

✅ Cohort Retrieval Complete!
   Found 15 datasets matching your criteria.
```

### Example 2: Missing Inputs
```
User: "Run pathway analysis"

🧠 Understanding Your Request
   Analyzing your query...

🎯 Routing to 🛤️ Pathway Enrichment Agent
   **Why this agent?**
   Keywords "pathway analysis" clearly indicate pathway enrichment.
   **Confidence:** 90%

📋 Checking Requirements
   Verifying required inputs...

📎 Additional Input Needed
   To run **Pathway Enrichment**, I need:
   
   📎 **prioritized_genes_path** - CSV file with prioritized genes
      Example: lupus_DEGs_prioritized.csv
   
   ✏️ **disease_name** - Disease name for scoring
      Example: lupus
   
   Please provide these and I'll continue with the analysis.
```

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=sk-...        # Required for intent routing
LANGCHAIN_API_KEY=...        # Optional: LangSmith tracing
```

### Customizing Agents

Add new agents by modifying `agent_registry.py`:

```python
NEW_AGENT = AgentInfo(
    agent_type=AgentType.NEW_AGENT,
    name="new_agent",
    display_name="🆕 New Agent",
    description="Description of what it does",
    required_inputs=[
        InputRequirement(
            name="input_name",
            description="What this input is",
            is_file=True,
            file_type="csv"
        )
    ],
    outputs=[
        OutputSpec(
            name="output_name",
            description="What this produces",
            state_key="output_key"
        )
    ],
    keywords=["keyword1", "keyword2"],
    example_queries=[
        "Example query 1",
        "Example query 2"
    ]
)

# Add to registry
AGENT_REGISTRY[AgentType.NEW_AGENT] = NEW_AGENT
```

## 📊 Status Update Types

The UI shows different status types with distinct styling:

| Type | Icon | Description |
|------|------|-------------|
| `THINKING` | 🧠 | Processing/understanding query |
| `ROUTING` | 🔍 | Determining which agent to use |
| `VALIDATING` | 📋 | Checking input requirements |
| `EXECUTING` | 🚀 | Running the agent |
| `PROGRESS` | 🔄 | Agent progress updates |
| `COMPLETED` | ✅ | Agent finished successfully |
| `ERROR` | ❌ | Something went wrong |
| `WAITING_INPUT` | 📎 | Need user to provide input |

## 🔄 Agent Dependencies

Some agents depend on outputs from others:
`
```
cohort_retrieval → deg_analysis → gene_prioritization → pathway_enrichment
                                                      
deconvolution (independent - can run anytime with bulk data)
```

The supervisor automatically tracks available outputs and suggests next steps.

## 📝 Logging

All operations are logged with structured formatting:

```
10:15:32 | INFO | 🧠 THINKING | Analyzing user query...
10:15:33 | INFO | 🎯 ROUTING | → pathway_enrichment (confidence: 92%)
10:15:33 | INFO |    └─ Reason: Keywords "pathway", "enrichment" detected...
10:15:33 | INFO | ✅ VALIDATING | Checking requirements
10:15:33 | WARNING |    └─ Missing inputs: prioritized_genes_path
10:15:33 | INFO | 📎 WAITING | Need user input: prioritized_genes_path
```

## 🧪 Testing

```bash
# Run full test suite
python -m agentic_ai_wf.supervisor_agent.test_supervisor

# Test specific components
python -c "
from agentic_ai_wf.supervisor_agent import AGENT_REGISTRY
for a in AGENT_REGISTRY.values():
    print(f'{a.display_name}: {a.description}')
"
```

## 🔮 Future Enhancements

- [ ] Multi-agent chaining (run full pipeline automatically)
- [ ] Result visualization in UI
- [ ] Export conversation history
- [ ] Agent performance metrics
- [ ] Custom agent plugins
