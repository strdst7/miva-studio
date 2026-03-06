MIVA Studio
Identity-Consistent Agentic AI

Runtime Guardrails • Deterministic Testing • Behavioral Evaluation

## Certification Context

This project was developed as part of the Production-Grade Agentic AI
Certification to demonstrate how an agentic system can be upgraded with
testing, runtime safety enforcement, and evaluation.

# MIVA-Studio

Production-Grade Identity-Consistent Agentic AI System

MIVA Studio is a production-oriented agentic AI system designed to maintain identity consistency in generative workflows. The project demonstrates how a working AI pipeline can be upgraded with runtime guardrails, deterministic testing, observability, and evaluation to achieve operational reliability.

Unlike many generative AI demos that prioritize novelty, MIVA Studio prioritizes behavioral stability and trust.

This repository serves as a capstone submission for a Production-Grade Agentic AI Certification, illustrating practical approaches to safety enforcement and regression monitoring in agentic systems.

⸻

Problem

Generative AI systems often produce convincing outputs but may exhibit behavioral instability in production environments.

In identity-critical workflows, small variations can lead to:
	•	identity drift across generations
	•	over-generation loops
	•	silent regression after model updates
	•	reduced user confidence in outputs

These issues are not always detectable through prompt design alone.

MIVA Studio addresses these risks by introducing runtime validation, deterministic testing, and structured evaluation.

⸻

System Architecture

The system is implemented as a stateful agent pipeline orchestrated with LangGraph. Generation, validation, and evaluation are separated into explicit components.

flowchart TD

A[User Input] --> B[LangGraph Agent]
B --> C[Generation Node]
C --> D[Identity Guardrail]

D -->|Pass| E[Accept Output]
D -->|Block| F[Regenerate]

E --> G[Metrics & Observability]
F --> G

G --> H[Ready Tensor Evaluation]

Architectural Principles

• Validation occurs after generation but before output acceptance
• Guardrails operate independently from prompt logic
• Failure paths are explicit and observable
• Evaluation occurs through comparative experiments

⸻

Key System Components

Agent Orchestration

LangGraph coordinates stateful workflow execution. Each generation cycle moves through defined nodes with conditional routing based on validation outcomes.

Identity Guardrails

A runtime guardrail evaluates identity drift using embedding similarity between:
	•	stored identity anchor
	•	generated output representation

If the drift score exceeds a configurable threshold, the output is rejected and regeneration is triggered.

Observability

Behavioral signals are logged during execution:
	•	identity drift score
	•	guardrail trigger rate
	•	regeneration attempts
	•	discard ratio

These signals provide insight into safety-performance tradeoffs.

Evaluation Framework

Experiments are conducted using Ready Tensor to compare system behavior across configurations.

⸻

Runtime Safety Enforcement

Safety validation is implemented as a dedicated guardrail node.

Guardrails evaluate:
	•	embedding similarity between anchor and candidate
	•	configurable drift threshold
	•	explicit pass / block decisions

Unlike prompt-based constraints, this approach enforces safety during runtime execution.

Example output:

GuardrailResult(
  passed=False,
  reason="identity_drift",
  drift_score=0.41,
  threshold=0.25
)


⸻

Testing Strategy

Testing is implemented using Pytest.

To ensure deterministic results:
	•	LLM calls are mocked
	•	node behavior is tested independently
	•	graph routing behavior is verified

Test coverage includes:

• identity guardrail pass/fail logic
• threshold parameterization
• missing state handling
• regeneration routing
• integration tests across nodes

Run tests with:

pytest


⸻

Evaluation Framework

The system is evaluated using Ready Tensor experiments comparing three configurations.

Configurations
	1.	Baseline (no guardrails)
	2.	Guardrails enabled
	3.	Threshold-tuned guardrails

Evaluation Flow

flowchart LR

A[Input Dataset] --> B[Run Baseline System]
B --> C[Collect Metrics]
C --> D[Baseline Report]

A --> E[Run Guardrail System]
E --> F[Collect Metrics]
F --> G[Guardrail Report]

D --> H[Comparative Analysis]
G --> H

H --> I[Regression Detection]


⸻

Behavioral Metrics

Metric	Purpose
identity_drift_score	measures output consistency
guardrail_trigger_rate	indicates safety enforcement
regeneration_count	detects over-generation loops
discard_ratio	proxy for user confidence

These metrics evaluate system behavior rather than aesthetic output quality.

⸻

Repository Structure

code/
  consts.py
  display_utils.py
  langgraph_utils.py
  identity_guardrails.py
  identity_bias_scan.py
  llm.py
  paths.py
  prompt_builder.py
  run_miva_studio.py
  utils.py

  graphs/
    miva_identity_graph.py

  nodes/
    miva_nodes.py
    node_utils.py
    output_types.py

  states/
    miva_identity_state.py

config/
  config.yaml
  reasoning.yaml
  gazetteer_entities.yaml

data/
  publication_examples/

evaluation/
  metrics.py
  experiments/

tests/
  unit/
  integration/


⸻

Installation

Clone the repository

git clone https://github.com/your-username/miva-studio

Install dependencies

pip install -r requirements.txt


⸻

Running the System

python code/run_miva_studio.py


⸻

Running Tests

pytest


⸻

Limitations

Current limitations include:

• identity validation assumes single-subject outputs
• embedding similarity depends on embedding model stability
• strict thresholds may reduce creative flexibility

These tradeoffs are documented and tunable through evaluation experiments.

⸻

Certification Context

This project was developed to demonstrate how an agentic AI system can be upgraded to production-grade reliability by integrating:
	•	runtime guardrails
	•	deterministic testing
	•	behavioral observability
	•	structured evaluation

The goal is operational credibility rather than demonstration capability.

⸻

License

MIT License

⸻

Author

Nur Amirah Mohd Kamil

⸻

Final Note

MIVA Studio demonstrates that production readiness in agentic AI requires more than generation capability. Reliable systems must include explicit safety enforcement, measurable evaluation, and deterministic validation mechanisms.
