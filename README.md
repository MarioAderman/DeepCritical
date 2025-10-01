# DeepCritical - Hydra + Pydantic Graph Deep Research with PRIME Architecture

A comprehensive research automation platform that replicates the PRIME (Protein Research Intelligent Multi-Agent Environment) architecture for autonomous scientific discovery workflows.

## 🚀 Quickstart

### Using uv (Recommended)

```bash
# Install uv and dependencies
uv sync

# Single REACT mode
uv run deepresearch question="What is machine learning?" app_mode=single_react

# Multi-level REACT with nested loops
uv run deepresearch question="Analyze machine learning in drug discovery" app_mode=multi_level_react

# Complex nested orchestration
uv run deepresearch question="Design a comprehensive research framework" app_mode=nested_orchestration

# Loss-driven execution
uv run deepresearch question="Optimize research quality" app_mode=loss_driven

# Using configuration files
uv run deepresearch --config-name=config_with_modes question="Your question" app_mode=multi_level_react
```

### Using pip (Legacy)

```bash
# Single REACT mode
deepresearch question="What is machine learning?" app_mode=single_react

# Multi-level REACT with nested loops
deepresearch question="Analyze machine learning in drug discovery" app_mode=multi_level_react

# Complex nested orchestration
deepresearch question="Design a comprehensive research framework" app_mode=nested_orchestration

# Loss-driven execution
deepresearch question="Optimize research quality" app_mode=loss_driven

# Using configuration files
deepresearch --config-name=config_with_modes question="Your question" app_mode=multi_level_react
```

### 1) Installation

#### Using uv (Recommended)

```bash
# Install uv if not already installed
# Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Run the application
uv run deepresearch --help
```

#### Using pip (Alternative)

```bash
# Create virtual environment
python -m venv .venv && .venv\Scripts\activate

# Install package
pip install -e .
```

### 2) Basic Usage

#### Using uv (Recommended)

```bash
# Run default workflow
uv run deepresearch

# Run with custom question
uv run deepresearch question="What are PRIME's core contributions?"

# Run with specific configuration
uv run deepresearch --config-name=config_with_modes question="Your question" app_mode=multi_level_react
```

#### Using pip (Alternative)

```bash
# Run default workflow
python -m deepresearch.app

# Run with custom question
python -m deepresearch.app question="What are PRIME's core contributions?"
```

### 3) PRIME Flow (Protein Engineering)

#### Using uv (Recommended)

```bash
# Design therapeutic antibody
uv run deepresearch flows.prime.enabled=true question="Design a therapeutic antibody for SARS-CoV-2"

# Analyze protein sequence
uv run deepresearch flows.prime.enabled=true question="Analyze protein sequence MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# Predict protein structure
uv run deepresearch flows.prime.enabled=true question="Predict 3D structure of protein P12345"
```

#### Using pip (Alternative)

```bash
# Design therapeutic antibody
python -m deepresearch.app flows.prime.enabled=true question="Design a therapeutic antibody for SARS-CoV-2"

# Analyze protein sequence
python -m deepresearch.app flows.prime.enabled=true question="Analyze protein sequence MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"

# Predict protein structure
python -m deepresearch.app flows.prime.enabled=true question="Predict 3D structure of protein P12345"
```

### 4) Bioinformatics Flow (Data Fusion & Reasoning)

```bash
# GO + PubMed reasoning for gene function
python -m deepresearch.app flows.bioinformatics.enabled=true question="What is the function of TP53 gene based on GO annotations and recent literature?"

# Multi-source drug-target analysis
python -m deepresearch.app flows.bioinformatics.enabled=true question="Analyze the relationship between drug X and protein Y using expression profiles and interactions"

# Protein structure-function analysis
python -m deepresearch.app flows.bioinformatics.enabled=true question="What is the likely function of protein P12345 based on its structure and GO annotations?"
```

### 5) Flow Selection

```bash
# PRIME flow (protein engineering)
python -m deepresearch.app flows.prime.enabled=true

# Bioinformatics flow (data fusion & reasoning)
python -m deepresearch.app flows.bioinformatics.enabled=true

# DeepSearch flow (web research)
python -m deepresearch.app flows.deepsearch.enabled=true

# Challenge flow (experimental)
python -m deepresearch.app challenge.enabled=true
```

### 6) Advanced Configuration

```bash
# Custom plan steps
python -m deepresearch.app plan='["clarify scope","collect sources","synthesize"]'

# Manual confirmation mode
python -m deepresearch.app flows.prime.params.manual_confirmation=true

# Disable adaptive re-planning
python -m deepresearch.app flows.prime.params.adaptive_replanning=false
```

⚠️ **Known Issues:**
- Circular import issues in some tool modules (bioinformatics_tools, deep_agent_tools)
- Some pydantic-ai API compatibility issues (defer decorator not available in current version)
- These issues are being addressed and will be resolved in future updates

## 🏗️ Architecture

### Core Components

- **Hydra Configuration**: Uses Hydra composition for configuration (`configs/`) per [Hydra docs](https://hydra.cc/docs/intro/)
- **Pydantic Graph**: Stateful workflow execution (`deepresearch/app.py`) per [Pydantic Graph docs](https://ai.pydantic.dev/graph/#stateful-graphs)
- **PRIME Integration**: Replicates the PRIME paper's three-stage architecture

### PRIME Three-Stage Architecture

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│  Parse  │───▶│  Plan   │───▶│ Execute │
│         │    │         │    │         │
│ Query   │    │ DAG     │    │ Tool    │
│ Parser  │    │ Gen.    │    │ Exec.   │
└─────────┘    └─────────┘    └─────────┘
```

1. **Parse** → `QueryParser` - Semantic/syntactic analysis of research queries
2. **Plan** → `PlanGenerator` - DAG workflow construction with 65+ tools  
3. **Execute** → `ToolExecutor` - Adaptive re-planning with strategic/tactical recovery

## 🧬 PRIME Features

### Protein Engineering Tool Ecosystem
- **65+ Tools** across 6 categories: Knowledge Query, Sequence Analysis, Structure Prediction, Molecular Docking, De Novo Design, Function Prediction
- **Scientific Intent Detection**: Automatically categorizes queries (protein_design, binding_analysis, structure_prediction, etc.)
- **Domain-Specific Heuristics**: Immunology, enzymology, cell biology, general protein domains

### Adaptive Re-planning System
- **Strategic Re-planning**: Tool substitution (BLAST → ProTrek, AlphaFold2 → ESMFold)
- **Tactical Re-planning**: Parameter adjustment (E-value relaxation, exhaustiveness tuning)
- **Execution History**: Comprehensive tracking with failure pattern analysis
- **Success Criteria Validation**: Quantitative metrics (pLDDT, E-values) and binary outcomes

### Scientific Grounding
- **Verifiable Results**: All conclusions come from validated tools, never from LLM generation
- **Tool Validation**: Strict input/output schema compliance and type checking
- **Mock Implementations**: Complete development environment with realistic outputs
- **Error Recovery**: Graceful handling with actionable recommendations

## 🧬 Bioinformatics Integration

### Multi-Source Data Fusion
- **GO + PubMed**: Gene Ontology annotations with paper context for reasoning tasks
- **GEO + CMAP**: Gene expression data with perturbation profiles
- **DrugBank + TTD + CMAP**: Drug-target-perturbation relationship graphs
- **PDB + IntAct**: Protein structure-interaction datasets

### Agent-to-Agent Communication
- **Specialized Agents**: DataFusionAgent, GOAnnotationAgent, ReasoningAgent, DataQualityAgent
- **Pydantic AI Integration**: Multi-model reasoning with evidence integration
- **Deferred Tools**: Efficient data processing with registry integration
- **Quality Assessment**: Cross-database consistency and evidence validation

### Integrative Reasoning
- **Non-Reductionist Approach**: Multi-source evidence integration beyond structural similarity
- **Evidence Code Prioritization**: IDA (gold standard) > EXP > computational predictions
- **Cross-Database Validation**: Consistency checks and temporal relevance
- **Human Curation Integration**: Leverages existing curation expertise

### Example Data Fusion
```json
{
  "pmid": "12345678",
  "title": "p53 mediates the DNA damage response in mammalian cells",
  "abstract": "DNA damage induces p53 stabilization, leading to cell cycle arrest and apoptosis.",
  "gene_id": "P04637",
  "gene_symbol": "TP53",
  "go_term_id": "GO:0006977",
  "go_term_name": "DNA damage response",
  "evidence_code": "IDA",
  "annotation_note": "Curated based on experimental results in Figure 3."
}
```

## 🔄 Flow Architecture

### Available Flows
- **PRIME Flow**: Protein engineering with 65+ specialized tools
- **Bioinformatics Flow**: Multi-source data fusion and integrative reasoning
- **DeepSearch Flow**: Web research and information gathering
- **Challenge Flow**: Experimental workflows for research challenges
- **Default Flow**: General-purpose research automation

### Flow Orchestration
```
Plan → Route to Flow → Execute Subflow → Synthesize Results
  │
  ├─ PRIME: Parse → Plan → Execute → Evaluate
  ├─ Bioinformatics: Parse → Fuse → Assess → Reason → Synthesize
  ├─ DeepSearch: DSPlan → DSExecute → DSAnalyze → DSSynthesize  
  └─ Challenge: PrepareChallenge → RunChallenge → EvaluateChallenge
```

## ⚙️ Configuration

### Main Configuration

Key parameters in `configs/config.yaml`:

```yaml
# Research parameters
question: "Your research question here"
plan: ["step1", "step2", "step3"]
retries: 3
manual_confirm: false

# Flow control
flows:
  prime:
    enabled: true
    params:
      adaptive_replanning: true
      manual_confirmation: false
      tool_validation: true
  bioinformatics:
    enabled: true
    data_sources:
      go:
        enabled: true
        evidence_codes: ["IDA", "EXP"]
        year_min: 2022
        quality_threshold: 0.9
      pubmed:
        enabled: true
        max_results: 50
        include_full_text: true
    fusion:
      quality_threshold: 0.85
      max_entities: 500
      cross_reference_enabled: true
    reasoning:
      model: "anthropic:claude-sonnet-4-0"
      confidence_threshold: 0.8
      integrative_approach: true

# Output management
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### Flow-Specific Configuration

Each flow has its own configuration file:

- `configs/statemachines/flows/prime.yaml` - PRIME flow parameters
- `configs/statemachines/flows/bioinformatics.yaml` - Bioinformatics flow parameters
- `configs/statemachines/flows/deepsearch.yaml` - DeepSearch parameters  
- `configs/statemachines/flows/hypothesis_generation.yaml` - Hypothesis flow
- `configs/statemachines/flows/execution.yaml` - Execution flow
- `configs/statemachines/flows/reporting.yaml` - Reporting flow

### Prompt Configuration

Prompt templates in `configs/prompts/`:

- `configs/prompts/prime_parser.yaml` - Query parsing prompts
- `configs/prompts/prime_planner.yaml` - Workflow planning prompts
- `configs/prompts/prime_executor.yaml` - Tool execution prompts
- `configs/prompts/prime_evaluator.yaml` - Result evaluation prompts

## 🔧 Development

### Development with uv

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .

# Add new dependencies
uv add package_name

# Add development dependencies
uv add --dev package_name

# Update dependencies
uv lock --upgrade

# Run scripts in the project environment
uv run python script.py
```

### Project Structure

```
DeepCritical/
├── deepresearch/           # Main package
│   ├── app.py             # Pydantic Graph workflow
│   ├── src/               # PRIME implementation
│   │   ├── agents/        # PRIME agents (Parser, Planner, Executor)
│   │   ├── datatypes/     # Bioinformatics data types
│   │   ├── statemachines/ # Bioinformatics workflows
│   │   └── utils/         # Utilities (Tool Registry, Execution History)
│   └── tools/             # Tool implementations
├── configs/               # Hydra configuration
│   ├── config.yaml        # Main configuration
│   ├── prompts/           # Prompt templates
│   └── statemachines/     # Flow configurations
├── docs/                  # Documentation
│   └── bioinformatics_integration.md
└── .cursor/rules/         # Cursor rules for development
```

### Extending Flows

1. **Create Flow Configuration**:
   ```yaml
   # configs/statemachines/flows/my_flow.yaml
   enabled: true
   params:
     custom_param: "value"
   ```

2. **Implement Nodes**:
   ```python
   @dataclass
   class MyFlowNode(BaseNode[ResearchState]):
       async def run(self, ctx: GraphRunContext[ResearchState]) -> NextNode:
           # Implementation
           return NextNode()
   ```

3. **Register in Graph**:
   ```python
   # In run_graph function
   nodes = (..., MyFlowNode())
   ```

4. **Add Flow Routing**:
   ```python
   # In Plan node
   if getattr(cfg.flows, "my_flow", {}).get("enabled"):
       return MyFlowNode()
   ```

### Tool Development

1. **Define Tool Specification**:
   ```python
   tool_spec = ToolSpec(
       name="my_tool",
       category=ToolCategory.SEQUENCE_ANALYSIS,
       input_schema={"sequence": "string"},
       output_schema={"result": "dict"},
       success_criteria={"min_confidence": 0.8}
   )
   ```

2. **Implement Tool Runner**:
   ```python
   class MyToolRunner(ToolRunner):
       def run(self, parameters: Dict[str, Any]) -> ExecutionResult:
           # Tool implementation
           return ExecutionResult(success=True, data=result)
   ```

3. **Register Tool**:
   ```python
   registry.register_tool(tool_spec, MyToolRunner)
   ```

### Bioinformatics Development

1. **Create Data Types**:
   ```python
   from pydantic import BaseModel, Field
   
   class GOAnnotation(BaseModel):
       pmid: str = Field(..., description="PubMed ID")
       gene_id: str = Field(..., description="Gene identifier")
       go_term: GOTerm = Field(..., description="GO term")
       evidence_code: EvidenceCode = Field(..., description="Evidence code")
   ```

2. **Implement Agents**:
   ```python
   from pydantic_ai import Agent
   
   class DataFusionAgent:
       def __init__(self, model_name: str):
           self.agent = Agent(
               model=AnthropicModel(model_name),
               deps_type=BioinformaticsAgentDeps,
               result_type=DataFusionResult
           )
   ```

3. **Create Workflow Nodes**:
   ```python
   @dataclass
   class FuseDataSources(BaseNode[BioinformaticsState]):
       async def run(self, ctx: GraphRunContext[BioinformaticsState]) -> NextNode:
           # Data fusion logic
           return AssessDataQuality()
   ```

4. **Register Deferred Tools**:
   ```python
   from pydantic_ai.tools import defer
   
   @defer
   def go_annotation_processor(annotations, papers, evidence_codes):
       # Processing logic
       return processed_annotations
   ```

## 🚀 Advanced Usage

### Batch Processing

```bash
# Run multiple experiments
python -m deepresearch.app --multirun \
  question="Design antibody for SARS-CoV-2",question="Analyze protein P12345" \
  flows.prime.enabled=true
```

### Custom Tool Integration

```python
from deepresearch.src.utils.tool_registry import ToolRegistry, ToolSpec, ToolCategory

# Create custom tool
registry = ToolRegistry()
tool_spec = ToolSpec(
    name="custom_analyzer",
    category=ToolCategory.SEQUENCE_ANALYSIS,
    input_schema={"sequence": "string"},
    output_schema={"analysis": "dict"}
)
registry.register_tool(tool_spec)
```

### Execution History Analysis

```python
from deepresearch.src.utils.execution_history import ExecutionHistory

# Load execution history
history = ExecutionHistory.load_from_file("outputs/2024-01-01/12-00-00/execution_history.json")

# Analyze performance
summary = history.get_execution_summary()
print(f"Success rate: {summary['success_rate']:.2%}")
print(f"Tools used: {summary['tools_used']}")
```

## 📚 References

- [Hydra Documentation](https://hydra.cc/docs/intro/) - Configuration management
- [Pydantic Graph](https://ai.pydantic.dev/graph/#stateful-graphs) - Stateful workflow execution
- [Pydantic AI](https://ai.pydantic.dev/) - Agent-to-agent communication
- [PRIME Paper](https://doi.org/10.1101/2025.09.22.677756) - Original research paper
- [Bioinformatics Integration](docs/bioinformatics_integration.md) - Multi-source data fusion guide
- [Protein Engineering Tools](https://github.com/facebookresearch/hydra) - Tool ecosystem reference

