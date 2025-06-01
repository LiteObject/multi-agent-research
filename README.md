# Multi-Agent Research

A Streamlit app for running a multi-agent health science research workflow.

## Features
- Configure and launch multi-agent LLM workflows for health science research
- Real-time progress and agent status display
- Downloadable markdown reports
- Trusted sources configuration
- Supports OpenAI and Ollama LLMs

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
4. Run the app:
   ```sh
   streamlit run app.py
   ```

## Usage
- Configure workflow parameters in the sidebar
- Start/stop the workflow and monitor progress
- Download generated reports from the UI

## File Structure
- `app.py` — Main Streamlit app
- `multi_agent_workflow.py` — Workflow logic
- `llm_factory.py` — LLM integration
- `docs/` — Generated reports

## License
MIT License