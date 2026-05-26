import asyncio
import os
import datetime
import logging
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, List, Optional
from dataclasses import dataclass, field
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentWorkflow
from llm_factory import get_llm, get_embedding_model, LLMType
from research_profiles import (
    DEFAULT_PRIMARY_SOURCES,
    DEFAULT_SECONDARY_SOURCES,
    ResearchProfile,
    build_default_research_profile,
    get_effective_research_profile,
)
from workflow_adapters import (
    LocalReportPersistenceAdapter,
    ReportPersistenceAdapter,
    SearchToolAdapter,
    TavilySearchAdapter,
)

PRIMARY_SOURCE_HINTS = (
    "pubmed",
    "thelancet",
    "nature medicine",
    "nature.com",
    "sciencedirect",
    "nih.gov",
    "world health organization",
    "who.int",
    "cdc.gov",
    "medlineplus",
)


DEFAULT_TARGET_WORD_COUNT = 5000
DEFAULT_MIN_DEVELOPMENTS = 5
DEFAULT_MAX_DEVELOPMENTS = 7
DEFAULT_REPORT_FILENAME = "report.md"


def get_default_research_profile() -> ResearchProfile:
    """Return the active research profile with environment overrides applied."""
    return get_effective_research_profile(os.getenv("RESEARCH_PROFILE_PATH"))


def get_default_primary_sources() -> List[str]:
    """Return the default primary sources for the active profile."""
    return list(get_default_research_profile().primary_sources)


def get_default_secondary_sources() -> List[str]:
    """Return the default secondary sources for the active profile."""
    return list(get_default_research_profile().secondary_sources)


@dataclass
class WorkflowConfig:
    """Configuration class for the multi-agent workflow"""

    research_profile_path: Optional[str] = field(
        default_factory=lambda: os.getenv("RESEARCH_PROFILE_PATH")
    )

    research_profile: ResearchProfile = field(init=False, repr=False)

    # API Configuration
    tavily_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY")
    )

    # LLM Configuration
    llm_type: LLMType = LLMType.OPENAI
    llm_model: Optional[str] = None

    # File and Directory Configuration
    docs_dir: str = "./docs"
    default_report_filename: str = DEFAULT_REPORT_FILENAME

    # Report Configuration
    target_word_count: int = DEFAULT_TARGET_WORD_COUNT
    min_developments: int = DEFAULT_MIN_DEVELOPMENTS
    max_developments: int = DEFAULT_MAX_DEVELOPMENTS

    # Workflow Configuration
    max_iterations: int = 10
    timeout_seconds: int = 1800  # 30 minutes

    # Trusted Sources
    primary_sources: List[str] = field(default_factory=list)
    secondary_sources: List[str] = field(default_factory=list)
    trusted_sources: List[str] = field(default_factory=lambda: [])

    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Validate configuration after initialization"""
        self._initialize_research_profile()
        self._initialize_source_tiers()
        self.validate()
        self._setup_logging()
        self._ensure_directories()

    def _initialize_research_profile(self) -> None:
        """Load the active research profile and apply profile defaults."""
        self.research_profile = get_effective_research_profile(
            self.research_profile_path
        )

        if self.target_word_count == DEFAULT_TARGET_WORD_COUNT:
            self.target_word_count = self.research_profile.target_word_count

        if self.min_developments == DEFAULT_MIN_DEVELOPMENTS:
            self.min_developments = self.research_profile.min_developments

        if self.max_developments == DEFAULT_MAX_DEVELOPMENTS:
            self.max_developments = self.research_profile.max_developments

        if self.default_report_filename == DEFAULT_REPORT_FILENAME:
            self.default_report_filename = self.research_profile.default_report_filename

    def _initialize_source_tiers(self) -> None:
        """Normalize source tier inputs and keep the combined source list in sync."""
        if not self.primary_sources and not self.secondary_sources:
            if self.trusted_sources:
                self.primary_sources, self.secondary_sources = (
                    self._split_sources_by_priority(self.trusted_sources)
                )
            else:
                self.primary_sources = list(self.research_profile.primary_sources)
                self.secondary_sources = list(self.research_profile.secondary_sources)

        self.trusted_sources = [*self.primary_sources, *self.secondary_sources]

    def _normalize_source(self, source: str) -> str:
        return source.strip().lower()

    def is_primary_source(self, source: str) -> bool:
        normalized = self._normalize_source(source)
        return any(hint in normalized for hint in PRIMARY_SOURCE_HINTS)

    def _split_sources_by_priority(self, sources: List[str]):
        primary_sources = []
        secondary_sources = []

        for source in sources:
            if self.is_primary_source(source):
                primary_sources.append(source)
            else:
                secondary_sources.append(source)

        return primary_sources, secondary_sources

    def get_primary_trusted_sources(self) -> List[str]:
        """Return sources that should be preferred for factual claims."""
        return list(self.primary_sources)

    def get_secondary_trusted_sources(self) -> List[str]:
        """Return sources that should be used for context and lead generation."""
        return list(self.secondary_sources)

    def get_source_guidance(self) -> str:
        """Return a tiered source guidance block for prompts and instructions."""
        return self.research_profile.build_source_guidance(
            self.get_primary_trusted_sources(), self.get_secondary_trusted_sources()
        )

    def validate(self) -> None:
        """Validate configuration values"""
        errors = []

        if not self.tavily_api_key:
            errors.append("TAVILY_API_KEY environment variable is required")

        if self.target_word_count < 1000:
            errors.append(
                f"target_word_count must be at least 1000, got {self.target_word_count}"
            )

        if self.min_developments < 1 or self.max_developments < self.min_developments:
            errors.append("Invalid development count configuration")

        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")

        if self.timeout_seconds < 1:
            errors.append("timeout_seconds must be positive")

        if not self.trusted_sources:
            errors.append("At least one trusted source must be configured")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()), format=self.log_format
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
        source_guidance = self.get_source_guidance()
        return self.research_profile.build_prompt(
            target_word_count=self.target_word_count,
            min_developments=self.min_developments,
            max_developments=self.max_developments,
            current_month_year=current_month_year,
            source_guidance=source_guidance,
        )


class ReportStatus(str, Enum):
    """Lifecycle stages for report generation."""

    RESEARCHING = "researching"
    DRAFT_READY = "draft_ready"
    CHANGES_REQUESTED = "changes_requested"
    APPROVED = "approved"
    PUBLISHED = "published"


class WorkflowEventType(str, Enum):
    """Structured runtime events emitted by the workflow engine."""

    AGENT_CHANGE = "agent_change"
    AGENT_OUTPUT = "agent_output"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMPLETION = "completion"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass
class WorkflowState:
    """Typed view over the workflow state stored in the agent context."""

    research_notes: dict[str, str] = field(default_factory=dict)
    draft_report_content: str = ""
    report_content: str = "Not written yet."
    draft_report_filename: str = "report.md"
    final_report_path: str = ""
    review: str = "Review required."
    review_feedback: str = ""
    review_approved: bool = False
    report_status: ReportStatus = ReportStatus.RESEARCHING
    iteration_count: int = 0

    @classmethod
    def from_raw(cls, raw_state: Any, default_report_filename: str) -> "WorkflowState":
        """Build a typed state object from the workflow context payload."""
        if isinstance(raw_state, cls):
            return raw_state

        if raw_state is None:
            raw_state = {}

        if not isinstance(raw_state, dict):
            raise TypeError("Workflow state must be a dictionary payload.")

        research_notes = raw_state.get("research_notes") or {}
        if not isinstance(research_notes, dict):
            raise TypeError("Workflow state research_notes must be a dictionary.")

        raw_status = raw_state.get("report_status", ReportStatus.RESEARCHING.value)
        try:
            report_status = (
                raw_status
                if isinstance(raw_status, ReportStatus)
                else ReportStatus(str(raw_status))
            )
        except ValueError:
            report_status = ReportStatus.RESEARCHING

        return cls(
            research_notes={
                str(title): str(notes) for title, notes in research_notes.items()
            },
            draft_report_content=str(raw_state.get("draft_report_content", "")),
            report_content=str(raw_state.get("report_content", "Not written yet.")),
            draft_report_filename=str(
                raw_state.get("draft_report_filename", default_report_filename)
            ),
            final_report_path=str(raw_state.get("final_report_path", "")),
            review=str(raw_state.get("review", "Review required.")),
            review_feedback=str(raw_state.get("review_feedback", "")),
            review_approved=bool(raw_state.get("review_approved", False)),
            report_status=report_status,
            iteration_count=int(raw_state.get("iteration_count", 0)),
        )

    def to_raw(self) -> dict[str, Any]:
        """Convert the typed state back into the payload stored by the agent workflow."""
        return {
            "research_notes": dict(self.research_notes),
            "draft_report_content": self.draft_report_content,
            "report_content": self.report_content,
            "draft_report_filename": self.draft_report_filename,
            "final_report_path": self.final_report_path,
            "review": self.review,
            "review_feedback": self.review_feedback,
            "review_approved": self.review_approved,
            "report_status": self.report_status.value,
            "iteration_count": self.iteration_count,
        }

    def record_notes(self, notes_title: str, notes: str) -> None:
        self.research_notes[notes_title] = notes

    def store_draft(self, report_content: str, filename: str) -> None:
        self.report_content = report_content
        self.draft_report_content = report_content
        self.draft_report_filename = filename
        self.report_status = ReportStatus.DRAFT_READY
        self.review_approved = False

    def request_changes(self, review: str) -> None:
        self.review = review
        self.review_feedback = review
        self.review_approved = False
        self.report_status = ReportStatus.CHANGES_REQUESTED

    def approve(self) -> None:
        self.review = "Approved"
        self.review_feedback = "Approved"
        self.review_approved = True
        self.report_status = ReportStatus.APPROVED

    def publish(self, filepath: str) -> None:
        self.final_report_path = filepath
        self.report_content = self.draft_report_content
        self.report_status = ReportStatus.PUBLISHED

    def record_iteration(self, iteration_count: int) -> None:
        self.iteration_count = iteration_count


@dataclass
class WorkflowRuntimeEvent:
    """Normalized workflow event payload consumed by the CLI runner and UI."""

    type: WorkflowEventType
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    agent: Optional[str] = None
    iteration: Optional[int] = None
    content: Optional[str] = None
    tool_calls: list[str] = field(default_factory=list)
    tool_name: Optional[str] = None
    arguments: dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None
    message: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        """Return a UI-friendly dictionary payload."""
        payload = {
            "type": self.type.value,
            "timestamp": self.timestamp,
        }

        if self.agent is not None:
            payload["agent"] = self.agent
        if self.iteration is not None:
            payload["iteration"] = self.iteration
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_calls:
            payload["tool_calls"] = list(self.tool_calls)
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.arguments:
            payload["arguments"] = dict(self.arguments)
        if self.output is not None:
            payload["output"] = self.output
        if self.message is not None:
            payload["message"] = self.message

        return payload


class MultiAgentWorkflow:
    """Main workflow class that uses configuration"""

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        search_adapter: Optional[SearchToolAdapter] = None,
        persistence_adapter: Optional[ReportPersistenceAdapter] = None,
    ):
        self.config = config or WorkflowConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_runtime_dependencies()
        self.search_adapter = search_adapter or TavilySearchAdapter(
            self.config.tavily_api_key
        )
        self.persistence_adapter = persistence_adapter or LocalReportPersistenceAdapter(
            self.config.docs_dir,
            self.config.default_report_filename,
        )
        self._setup_tools()
        self._setup_agents()
        self._setup_workflow()

    def _setup_runtime_dependencies(self) -> None:
        """Initialize runtime dependencies owned by this workflow instance."""
        try:
            self.embedding_model = get_embedding_model(llm_type=self.config.llm_type)
            self.llm = get_llm(
                llm_type=self.config.llm_type,
                model=self.config.llm_model,
            )
            self.logger.info(
                "Workflow runtime initialized with %s%s",
                self.config.llm_type,
                f" ({self.config.llm_model})" if self.config.llm_model else "",
            )
        except Exception as e:
            self.logger.error("Failed to initialize workflow runtime: %s", e)
            raise

    def _setup_tools(self) -> None:
        """Initialize tools with error handling"""
        try:
            tavily_tools = self.search_adapter.get_search_tools()
            self.search_web = tavily_tools[0]
            self.logger.info("Tools initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize tools: %s", e)
            raise

    async def _get_workflow_state(self, ctx: Context) -> WorkflowState:
        """Load the typed workflow state from the agent context."""
        raw_state = await ctx.get("state")
        return WorkflowState.from_raw(
            raw_state, default_report_filename=self.config.default_report_filename
        )

    async def _set_workflow_state(self, ctx: Context, state: WorkflowState) -> None:
        """Persist the typed workflow state back into the agent context."""
        await ctx.set("state", state.to_raw())

    def _create_initial_state(self) -> dict[str, Any]:
        """Build the initial workflow state payload."""
        return WorkflowState(
            draft_report_filename=self.config.default_report_filename
        ).to_raw()

    def _setup_agents(self) -> None:
        """Setup all agents with configuration"""
        self.research_agent = FunctionAgent(
            name="ResearchAgent",
            description="Useful for searching the web for information on a given topic and recording notes on the topic.",
            system_prompt=(
                "You are the ResearchAgent that can search the web for information on a given topic and record notes. "
                "Prefer primary sources over secondary sources, use Google Scholar only to discover underlying primary papers, "
                "and capture clear source titles, publication dates, and URLs in the notes. "
                "Once notes are recorded and you are satisfied, hand off control to the WriteAgent to draft the report."
            ),
            llm=self.llm,
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
                f"Use write_report to store a draft, then wait for the ReviewAgent. "
                f"If the ReviewAgent approves the draft, use publish_report to save the final markdown file."
            ),
            llm=self.llm,
            tools=[
                self._create_write_report_tool(),
                self._create_publish_report_tool(),
            ],
            can_handoff_to=["ReviewAgent", "ResearchAgent"],
        )

        self.review_agent = FunctionAgent(
            name="ReviewAgent",
            description="Useful for reviewing a report and providing feedback.",
            system_prompt=(
                "You are the ReviewAgent that reviews reports and provides feedback. "
                "Your feedback should either request specific changes with review_report or approve the current draft with approve_report. "
                "When the draft is ready, approve it and hand off control back to the WriteAgent so it can publish the final report."
            ),
            llm=self.llm,
            tools=[
                self._create_review_report_tool(),
                self._create_approve_report_tool(),
            ],
            can_handoff_to=["WriteAgent"],
        )

        self.logger.info("All agents initialized successfully")

    def _create_record_notes_tool(self):
        """Create the record_notes tool with proper error handling"""

        async def record_notes(ctx: Context, notes: str, notes_title: str) -> str:
            try:
                state = await self._get_workflow_state(ctx)
                state.record_notes(notes_title=notes_title, notes=notes)
                await self._set_workflow_state(ctx, state)
                self.logger.info("Notes recorded: %s", notes_title)
                return "Notes recorded successfully."
            except (KeyError, TypeError, RuntimeError) as e:
                error_msg = f"Error recording notes: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

        return record_notes

    def _create_write_report_tool(self):
        """Create the write_report tool with configuration"""

        async def write_report(
            ctx: Context, report_content: str, filename: Optional[str] = None
        ) -> str:
            try:
                state = await self._get_workflow_state(ctx)
                state.store_draft(
                    report_content=report_content,
                    filename=filename or self.config.default_report_filename,
                )
                await self._set_workflow_state(ctx, state)

                self.logger.info("Draft report stored in workflow state")
                return "Draft report stored. Awaiting review before publishing."
            except (OSError, IOError) as file_error:
                error_msg = f"File error writing draft report: {str(file_error)}"
                self.logger.error(error_msg)
                return error_msg
            except (KeyError, TypeError) as state_error:
                error_msg = f"State error writing draft report: {str(state_error)}"
                self.logger.error(error_msg)
                return error_msg

        return write_report

    def _create_publish_report_tool(self):
        """Create the publish_report tool with approval gating"""

        async def publish_report(ctx: Context, filename: Optional[str] = None) -> str:
            try:
                state = await self._get_workflow_state(ctx)

                if not state.review_approved:
                    return "Cannot publish report until the ReviewAgent approves the draft."

                report_content = state.draft_report_content
                if not report_content:
                    return (
                        "Cannot publish report because no draft content is available."
                    )

                if (
                    state.report_status == ReportStatus.PUBLISHED
                    and state.final_report_path
                ):
                    return f"Report already published to {state.final_report_path}."

                filepath = self.persistence_adapter.publish_report(
                    report_content, filename or state.draft_report_filename
                )

                state.publish(filepath)
                await self._set_workflow_state(ctx, state)

                self.logger.info("Final report published to: %s", filepath)
                return f"Final report published to {filepath}."
            except (OSError, IOError) as file_error:
                error_msg = f"File error publishing report: {str(file_error)}"
                self.logger.error(error_msg)
                return error_msg
            except (KeyError, TypeError) as state_error:
                error_msg = f"State error publishing report: {str(state_error)}"
                self.logger.error(error_msg)
                return error_msg

        return publish_report

    def _create_review_report_tool(self):
        """Create the review_report tool"""

        async def review_report(ctx: Context, review: str) -> str:
            try:
                state = await self._get_workflow_state(ctx)
                state.request_changes(review)
                await self._set_workflow_state(ctx, state)
                self.logger.info("Review feedback recorded")
                return "Review feedback recorded successfully."
            except (KeyError, TypeError, RuntimeError) as e:
                error_msg = f"Error reviewing report: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

        return review_report

    def _create_approve_report_tool(self):
        """Create the approve_report tool"""

        async def approve_report(ctx: Context) -> str:
            try:
                state = await self._get_workflow_state(ctx)
                state.approve()
                await self._set_workflow_state(ctx, state)
                self.logger.info("Report approved")
                return "Report approved successfully."
            except (KeyError, TypeError, RuntimeError) as e:
                error_msg = f"Error approving report: {str(e)}"
                self.logger.error(error_msg)
                return error_msg

        return approve_report

    def _setup_workflow(self) -> None:
        """Setup the agent workflow"""
        self.agent_workflow = AgentWorkflow(
            agents=[self.research_agent, self.write_agent, self.review_agent],
            root_agent=self.research_agent.name,
            initial_state=self._create_initial_state(),
        )
        self.logger.info("Workflow initialized successfully")

    async def stream_runtime_events(
        self, cancel_requested: Optional[Callable[[], bool]] = None
    ) -> AsyncIterator[WorkflowRuntimeEvent]:
        """Yield normalized runtime events for the workflow execution."""
        self.logger.info("Starting multi-agent workflow")
        prompt = self.config.get_prompt_template()
        handler = self.agent_workflow.run(user_msg=prompt)

        current_agent = None
        iteration_count = 0
        event_iterator = handler.stream_events().__aiter__()
        loop = asyncio.get_running_loop()
        started_at = loop.time()

        while True:
            if cancel_requested and cancel_requested():
                self.logger.info("Workflow cancellation requested")
                yield WorkflowRuntimeEvent(
                    type=WorkflowEventType.CANCELLED,
                    message="Workflow cancelled by user.",
                )
                return

            elapsed = loop.time() - started_at
            remaining_timeout = self.config.timeout_seconds - elapsed
            if remaining_timeout <= 0:
                self.logger.warning(
                    "Workflow timed out after %s seconds",
                    self.config.timeout_seconds,
                )
                yield WorkflowRuntimeEvent(
                    type=WorkflowEventType.TIMED_OUT,
                    message=(
                        "Workflow timed out after "
                        f"{self.config.timeout_seconds} seconds."
                    ),
                )
                return

            try:
                event = await asyncio.wait_for(
                    event_iterator.__anext__(), timeout=remaining_timeout
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                self.logger.warning(
                    "Workflow timed out after %s seconds",
                    self.config.timeout_seconds,
                )
                yield WorkflowRuntimeEvent(
                    type=WorkflowEventType.TIMED_OUT,
                    message=(
                        "Workflow timed out after "
                        f"{self.config.timeout_seconds} seconds."
                    ),
                )
                return

            if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
            ):
                if iteration_count >= self.config.max_iterations:
                    self.logger.warning(
                        "Reached max iterations (%s)", self.config.max_iterations
                    )
                    yield WorkflowRuntimeEvent(
                        type=WorkflowEventType.COMPLETION,
                        message=(
                            "Workflow stopped after reaching max iterations "
                            f"({self.config.max_iterations})."
                        ),
                    )
                    return

                current_agent = event.current_agent_name
                iteration_count += 1

                yield WorkflowRuntimeEvent(
                    type=WorkflowEventType.AGENT_CHANGE,
                    agent=current_agent,
                    iteration=iteration_count,
                )

            elif isinstance(event, AgentOutput):
                if event.response.content or event.tool_calls:
                    yield WorkflowRuntimeEvent(
                        type=WorkflowEventType.AGENT_OUTPUT,
                        content=event.response.content,
                        tool_calls=(
                            [call.tool_name for call in event.tool_calls]
                            if event.tool_calls
                            else []
                        ),
                    )

            elif isinstance(event, ToolCallResult):
                yield WorkflowRuntimeEvent(
                    type=WorkflowEventType.TOOL_RESULT,
                    tool_name=event.tool_name,
                    arguments=getattr(event, "tool_kwargs", {}) or {},
                    output=str(event.tool_output),
                )

            elif isinstance(event, ToolCall):
                yield WorkflowRuntimeEvent(
                    type=WorkflowEventType.TOOL_CALL,
                    tool_name=event.tool_name,
                    arguments=event.tool_kwargs or {},
                )

        self.logger.info("Workflow completed successfully")
        yield WorkflowRuntimeEvent(
            type=WorkflowEventType.COMPLETION,
            message="Workflow completed successfully",
        )

    async def run(self) -> None:
        """Run the main workflow with error handling and monitoring"""
        try:
            async for event in self.stream_runtime_events():
                if event.type == WorkflowEventType.AGENT_CHANGE:
                    print(f"\n{'='*50}")
                    print(f"🤖 Agent: {event.agent} (Iteration {event.iteration})")
                    print(f"{'='*50}\n")
                    self.logger.info("Agent changed to: %s", event.agent)

                elif event.type == WorkflowEventType.AGENT_OUTPUT:
                    if event.content:
                        print("📤 Output:", event.content)
                    if event.tool_calls:
                        print(
                            "🛠️  Planning to use tools:",
                            event.tool_calls,
                        )

                elif event.type == WorkflowEventType.TOOL_RESULT:
                    print(f"🔧 Tool Result ({event.tool_name}):")
                    print(f"  Arguments: {event.arguments}")
                    print(f"  Output: {event.output}")

                elif event.type == WorkflowEventType.TOOL_CALL:
                    print(f"🔨 Calling Tool: {event.tool_name}")
                    print(f"  With arguments: {event.arguments}")

                elif (
                    event.type
                    in (
                        WorkflowEventType.COMPLETION,
                        WorkflowEventType.CANCELLED,
                        WorkflowEventType.TIMED_OUT,
                    )
                    and event.message
                ):
                    print(event.message)

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
