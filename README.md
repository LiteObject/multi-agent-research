# Multi-Agent Research

A Streamlit app for running a configurable multi-agent research workflow.

## Features
- Configure and launch multi-agent LLM workflows for topic-specific research
- Pick a research profile from the sidebar to switch topic defaults
- Real-time progress and agent status display
- Downloadable markdown reports
- Trusted sources configuration
- Supports OpenAI and Ollama LLMs

## Architecture
- `multi_agent_workflow.py` owns the workflow engine, typed workflow state, agent roles, and normalized runtime events.
- `research_profiles.py` loads topic-specific research profiles and composes the prompt source policy.
- `workflow_adapters.py` isolates Tavily search initialization and report persistence from the engine.
- `app.py` owns the Streamlit UI, run control, and event rendering.
- See [docs/agentic-architecture.md](docs/agentic-architecture.md) for the explicit workflow contract.

## Requirements
- Python 3.10+
- See `requirements.txt` for dependencies

## Setup
1. Clone the repository:
   ```sh
   git clone <repo-url>
   cd multi-agent-research
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your API keys as needed:
   - For Tavily (required for web search):
     ```env
     TAVILY_API_KEY=your_tavily_api_key_here
     ```
   - For OpenAI (required if you select OpenAI as LLM Type):
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - To switch the active research profile used by the workflow and UI defaults:
      ```env
      RESEARCH_PROFILE_PATH=research_profiles/health_science.yaml
      ```
      Profiles are YAML files stored under `research_profiles/`. Copy the default
      profile and adjust the topic, audience, tone, target word count, report filename,
      and source tiers to fit a different research domain.
   - To override the default research source tiers used by the workflow and sidebar:
      ```env
      PRIMARY_TRUSTED_SOURCES=PubMed (pubmed.ncbi.nlm.nih.gov),The Lancet (thelancet.com),Nature Medicine (nature.com/nm)
      SECONDARY_TRUSTED_SOURCES=Google Scholar,ScienceDaily (sciencedaily.com),Healthline (healthline.com)
      ```
      Values can be comma-, semicolon-, or newline-separated.
4. Run the app:
   ```sh
   streamlit run app.py
   ```

## Testing
- Run the workflow tests with:
   ```sh
   .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
   ```
- The mocked orchestration test covers the research -> draft -> approve -> publish path without calling external services.

## Usage
- Choose a research profile in the sidebar to control the topic, prompt defaults, and source tiers
- Configure workflow parameters in the sidebar
- Start/stop the workflow and monitor progress
- Download generated reports from the UI

## File Structure
- `app.py` — Main Streamlit app
- `multi_agent_workflow.py` — Workflow logic
- `workflow_adapters.py` — External service adapters for search and persistence
- `llm_factory.py` — LLM integration
- `tests/` — Config and orchestration tests
- `docs/` — Generated reports

## License
MIT License