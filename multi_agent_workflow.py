import os
import datetime
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core import Settings
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llm_factory import get_llm, get_embedding_model, LLMType


@dataclass
class WorkflowConfig:
    """Configuration class for the multi-agent workflow"""

    # API Configuration
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY"))

    # LLM Configuration
    llm_type: LLMType = LLMType.OPENAI

    # File and Directory Configuration
    docs_dir: str = "./docs"
    default_report_filename: str = "report.md"

    # Report Configuration
    target_word_count: int = 5000
    min_developments: int = 5
    max_developments: int = 7

    # Workflow Configuration
    max_iterations: int = 10
    timeout_seconds: int = 1800  # 30 minutes

    # Trusted Sources
    trusted_sources: List[str] = field(default_factory=lambda: [
        # Scientific Journals and Research Databases
        "PubMed (pubmed.ncbi.nlm.nih.gov)",
        "Google Scholar",
        "The Lancet (thelancet.com)",
        "Nature Medicine (nature.com/nm)",
        "ScienceDirect (sciencedirect.com)",  # <-- Added missing comma
        # Health News Aggregators and Research Summaries
        "ScienceDaily (sciencedaily.com)",
        "Medical News Today (medicalnewstoday.com)",
        "Healthline (healthline.com)",
        "Everyday Health (everydayhealth.com)",
        # Government and Institutional Health Websites
        "National Institutes of Health (NIH) (nih.gov)",
        "World Health Organization (WHO) (who.int)",
        "Centers for Disease Control and Prevention (CDC) (cdc.gov)",
        "MedlinePlus (medlineplus.gov)",
        # Health Blogs and Professional Networks
        "Harvard Health Blog (health.harvard.edu)",
        "Kaiser Health News (KHN) (kffhealthnews.org)",
        "WebMD Doctors Blog (webmd.com)",
        "MobiHealthNews (mobihealthnews.com)",
        # Conferences and Webinars
        "HIMSS (Healthcare Information and Management Systems Society) (himss.org)",
        "American Public Health Association (APHA) (apha.org)",
        "TEDMED (tedmed.com)"
    ])

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Validate configuration after initialization"""
        self.validate()
        self._setup_logging()
        self._ensure_directories()

    def validate(self) -> None:
        """Validate configuration values"""
        errors = []

        if not self.tavily_api_key:
            errors.append("TAVILY_API_KEY environment variable is required")

        if self.target_word_count < 1000:
            errors.append(
                f"target_word_count must be at least 1000, got {self.target_word_count}")

        if self.min_developments < 1 or self.max_developments < self.min_developments:
            errors.append("Invalid development count configuration")

        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")

        if errors:
            raise ValueError(
                f"Configuration validation failed: {'; '.join(errors)}")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format=self.log_format
        )

    def _ensure_directories(self) -> None:
        """Ensure required directories exist"""
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)

    def get_report_filepath(self, filename: Optional[str] = None) -> str:
        """Generate timestamped report filepath"""
        filename = filename or self.default_report_filename
        base, ext = os.path.splitext(filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_stamp = f"{base}_{timestamp}{ext}"
        return os.path.join(self.docs_dir, filename_with_stamp)

    def get_prompt_template(self) -> str:
        """Get the main prompt template with configuration values"""
        current_month_year = datetime.datetime.now().strftime("%B %Y")
        sources_str = ", ".join(self.trusted_sources)

        return f"""
        Write a {self.target_word_count}-word blog post that highlights and explains
        {self.min_developments} to {self.max_developments} of the most recent and significant
        developments in health science as of {current_month_year}. Use the most up-to-date
        information from trusted sources such as {sources_str}. For each development, cite
        your sources in-line in markdown format, including the article or study title,
        author(s) if available, publication date, and a direct URL. Focus on topics that
        are relevant to general readers and provide clear, accessible explanations of each
        breakthrough. Include relevant statistics, cite the source and publication date of
        each study or article, and end with a takeaway section summarizing why these updates
        matter for everyday health. Maintain a tone that is informative yet conversational,
        suitable for a health-conscious audience who may not have a medical background.
        """


class MultiAgentWorkflow:
    """Main workflow class that uses configuration"""

    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_llama_index()
        self._setup_tools()
        self._setup_agents()
        self._setup_workflow()

    def _setup_llama_index(self) -> None:
        """Initialize LlamaIndex settings"""
        try:
            Settings.embed_model = get_embedding_model(
                llm_type=self.config.llm_type)
            Settings.llm = get_llm(llm_type=self.config.llm_type)
            self.logger.info(
                "LlamaIndex initialized with %s", self.config.llm_type)
        except Exception as e:
            self.logger.error("Failed to initialize LlamaIndex: %s", e)
            raise

    def _setup_tools(self) -> None:
        """Initialize tools with error handling"""
        try:
            tavily_tool = TavilyToolSpec(api_key=self.config.tavily_api_key)
            tavily_tools = tavily_tool.to_tool_list()

            if not tavily_tools:
                raise ValueError("No Tavily tools available")

            self.search_web = tavily_tools[0]
            self.logger.info("Tools initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize tools: %s", e)
            raise

    def _setup_agents(self) -> None:
        """Setup all agents with configuration"""
        self.research_agent = FunctionAgent(
            name="ResearchAgent",
            description="Useful for searching the web for information on a given topic and recording notes on the topic.",
            system_prompt=(
                f"You are the ResearchAgent that can search the web for information on a given topic and record notes. "
                f"Focus on finding information from these trusted sources: {', '.join(self.config.trusted_sources)}. "
                f"Once notes are recorded and you are satisfied, hand off control to the WriteAgent to write a report."
            ),
            tools=[self.search_web, self._create_record_notes_tool()],
            can_handoff_to=["WriteAgent"],
        )

        self.write_agent = FunctionAgent(
            name="WriteAgent",
            description="Useful for writing a report on a given topic.",
            system_prompt=(
                f"You are the WriteAgent that writes reports in markdown format. "
                f"Target approximately {self.config.target_word_count} words. "
                f"Content should be grounded in research notes. "
                f"Once the report is written, get feedback from the ReviewAgent."
            ),
            tools=[self._create_write_report_tool()],
            can_handoff_to=["ReviewAgent", "ResearchAgent"],
        )

        self.review_agent = FunctionAgent(
            name="ReviewAgent",
            description="Useful for reviewing a report and providing feedback.",
            system_prompt=(
                "You are the ReviewAgent that reviews reports and provides feedback. "
                "Your feedback should either approve the current report or request specific changes "
                "for the WriteAgent to implement."
            ),
            tools=[self._create_review_report_tool()],
            can_handoff_to=["WriteAgent"],
        )

        self.logger.info("All agents initialized successfully")

    def _create_record_notes_tool(self):
        """Create the record_notes tool with proper error handling"""
        async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
            try:
                current_state = await ctx.get("state")
                if "research_notes" not in current_state:
                    current_state["research_notes"] = {}
                current_state["research_notes"][notes_title] = notes
                await ctx.set("state", current_state)
                self.logger.info("Notes recorded: %s", notes_title)
                return "Notes recorded successfully."
            except (KeyError, TypeError, RuntimeError) as e:
                error_msg = f"Error recording notes: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

        return record_notes

    def _create_write_report_tool(self):
        """Create the write_report tool with configuration"""
        async def write_report(ctx: Context, report_content: str, filename: Optional[str] = None) -> str:
            try:
                current_state = await ctx.get("state")
                current_state["report_content"] = report_content
                await ctx.set("state", current_state)

                filepath = self.config.get_report_filepath(filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(report_content)

                self.logger.info("Report written to: %s", filepath)
                return f"Report written and saved to {filepath}."
            except (OSError, IOError) as file_error:
                error_msg = f"File error writing report: {str(file_error)}"
                self.logger.error(error_msg)
                return error_msg
            except (KeyError, TypeError) as state_error:
                error_msg = f"State error writing report: {str(state_error)}"
                self.logger.error(error_msg)
                return error_msg

        return write_report

    def _create_review_report_tool(self):
        """Create the review_report tool"""
        async def review_report(ctx: Context, review: str) -> str:
            try:
                current_state = await ctx.get("state")
                current_state["review"] = review
                await ctx.set("state", current_state)
                self.logger.info("Report reviewed")
                return "Report reviewed successfully."
            except (KeyError, TypeError, RuntimeError) as e:
                error_msg = f"Error reviewing report: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

        return review_report

    def _setup_workflow(self) -> None:
        """Setup the agent workflow"""
        self.agent_workflow = AgentWorkflow(
            agents=[self.research_agent, self.write_agent, self.review_agent],
            root_agent=self.research_agent.name,
            initial_state={
                "research_notes": {},
                "report_content": "Not written yet.",
                "review": "Review required.",
                "iteration_count": 0,
            },
        )
        self.logger.info("Workflow initialized successfully")

    async def run(self) -> None:
        """Run the main workflow with error handling and monitoring"""
        try:
            self.logger.info("Starting multi-agent workflow")
            prompt = self.config.get_prompt_template()

            handler = self.agent_workflow.run(user_msg=prompt)

            current_agent = None
            iteration_count = 0

            async for event in handler.stream_events():
                # Check for timeout or max iterations
                if iteration_count >= self.config.max_iterations:
                    self.logger.warning(
                        "Reached max iterations (%s)", self.config.max_iterations)
                    break

                # Print when the current agent changes
                if (
                    hasattr(event, "current_agent_name")
                    and event.current_agent_name != current_agent
                ):
                    current_agent = event.current_agent_name
                    iteration_count += 1
                    print(f"\n{'='*50}")
                    print(
                        f"🤖 Agent: {current_agent} (Iteration {iteration_count})")
                    print(f"{'='*50}\n")
                    self.logger.info("Agent changed to: %s", current_agent)

                # Print agent output
                elif isinstance(event, AgentOutput):
                    if event.response.content:
                        print("📤 Output:", event.response.content)
                    if event.tool_calls:
                        print(
                            "🛠️  Planning to use tools:",
                            [call.tool_name for call in event.tool_calls],
                        )

                # Print tool call results
                elif isinstance(event, ToolCallResult):
                    print(f"🔧 Tool Result ({event.tool_name}):")
                    print(f"  Arguments: {event.tool_kwargs}")
                    print(f"  Output: {event.tool_output}")

                # Print when a tool is being called
                elif isinstance(event, ToolCall):
                    print(f"🔨 Calling Tool: {event.tool_name}")
                    print(f"  With arguments: {event.tool_kwargs}")

            self.logger.info("Workflow completed successfully")

        except Exception as e:
            self.logger.error("Workflow failed: %s", e)
            raise


# Usage examples and main function
async def main():
    """Main function demonstrating different configuration approaches"""

    # Option 1: Use default configuration
    print("=== Using Default Configuration ===")
    try:
        workflow1 = MultiAgentWorkflow()
        await workflow1.run()
    except (ValueError, OSError) as e:
        print(f"Error with default config: {e}")

    # Option 2: Custom configuration
    # print("\n=== Using Custom Configuration ===")
    # try:
    #     custom_config = WorkflowConfig(
    #         target_word_count=3000,
    #         min_developments=3,
    #         max_developments=5,
    #         docs_dir="./custom_reports",
    #         max_iterations=5,
    #         log_level="DEBUG",
    #         trusted_sources=["PubMed", "Nature", "Science", "NIH"]
    #     )
    #     workflow2 = MultiAgentWorkflow(custom_config)
    #     await workflow2.run()
    # except Exception as e:
    #     print(f"Error with custom config: {e}")

    # Option 3: Configuration from environment variables
    # print("\n=== Using Environment-Based Configuration ===")
    # try:
    #     # You could extend this to read more values from environment
    #     env_config = WorkflowConfig(
    #         target_word_count=int(os.getenv("REPORT_WORD_COUNT", "5000")),
    #         docs_dir=os.getenv("REPORTS_DIR", "./docs"),
    #         log_level=os.getenv("LOG_LEVEL", "INFO"),
    #     )
    #     workflow3 = MultiAgentWorkflow(env_config)
    #     await workflow3.run()
    # except Exception as e:
    #     print(f"Error with environment config: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
