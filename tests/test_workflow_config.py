import os
import asyncio
import logging
import tempfile
import unittest
from unittest.mock import patch

from multi_agent_workflow import (
    LLMType,
    MultiAgentWorkflow,
    ReportStatus,
    WorkflowConfig,
    WorkflowEventType,
    WorkflowRuntimeEvent,
    WorkflowState,
)


class FakeToolPlan:
    def __init__(self, tool_name: str):
        self.tool_name = tool_name


class FakeAgentOutputEvent:
    def __init__(self, content: str = "", tool_calls=None):
        self.response = type("Response", (), {"content": content})()
        self.tool_calls = tool_calls or []


class FakeToolCallEvent:
    def __init__(self, tool_name: str, tool_kwargs: dict):
        self.tool_name = tool_name
        self.tool_kwargs = tool_kwargs


class FakeToolCallResultEvent(FakeToolCallEvent):
    def __init__(self, tool_name: str, tool_kwargs: dict, tool_output: str):
        super().__init__(tool_name, tool_kwargs)
        self.tool_output = tool_output


class FakeAgentChangeEvent:
    def __init__(self, current_agent_name: str):
        self.current_agent_name = current_agent_name


class FakeHandler:
    def __init__(self, events):
        self._events = events

    async def stream_events(self):
        for event in self._events:
            yield event


class FakeAgentWorkflow:
    def __init__(self, events):
        self._events = events
        self.last_user_msg = None

    def run(self, user_msg: str):
        self.last_user_msg = user_msg
        return FakeHandler(self._events)


class FakeRuntimeConfig:
    def __init__(self, max_iterations: int = 3, timeout_seconds: float = 30):
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds

    def get_prompt_template(self) -> str:
        return "prompt"


class FakeSlowHandler:
    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds

    async def stream_events(self):
        await asyncio.sleep(self.delay_seconds)
        yield FakeAgentChangeEvent("ResearchAgent")


class FakeSlowAgentWorkflow:
    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds

    def run(self, user_msg: str):
        return FakeSlowHandler(self.delay_seconds)


class WorkflowConfigSourceTierTests(unittest.TestCase):
    def _build_config(self, docs_dir: str, **overrides) -> WorkflowConfig:
        params = {
            "tavily_api_key": "test-key",
            "docs_dir": docs_dir,
            "target_word_count": 1000,
            "min_developments": 1,
            "max_developments": 2,
            "max_iterations": 1,
        }
        params.update(overrides)
        return WorkflowConfig(**params)

    def test_default_sources_are_split_into_tiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(temp_dir)

            self.assertIn(
                "PubMed (pubmed.ncbi.nlm.nih.gov)",
                config.get_primary_trusted_sources(),
            )
            self.assertIn(
                "World Health Organization (WHO) (who.int)",
                config.get_primary_trusted_sources(),
            )
            self.assertIn("Google Scholar", config.get_secondary_trusted_sources())
            self.assertIn(
                "Medical News Today (medicalnewstoday.com)",
                config.get_secondary_trusted_sources(),
            )
            self.assertTrue(config.is_primary_source("The Lancet (thelancet.com)"))
            self.assertFalse(config.is_primary_source("Google Scholar"))
            self.assertEqual(
                config.trusted_sources,
                config.get_primary_trusted_sources()
                + config.get_secondary_trusted_sources(),
            )

    def test_custom_trusted_sources_are_classified_by_priority(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(
                temp_dir,
                trusted_sources=[
                    "PubMed (pubmed.ncbi.nlm.nih.gov)",
                    "Google Scholar",
                    "Medical News Today (medicalnewstoday.com)",
                    "WHO (who.int)",
                ],
            )

            self.assertEqual(
                config.get_primary_trusted_sources(),
                ["PubMed (pubmed.ncbi.nlm.nih.gov)", "WHO (who.int)"],
            )
            self.assertEqual(
                config.get_secondary_trusted_sources(),
                ["Google Scholar", "Medical News Today (medicalnewstoday.com)"],
            )

    def test_env_defaults_override_source_tiers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {
                    "PRIMARY_TRUSTED_SOURCES": (
                        "PubMed (pubmed.ncbi.nlm.nih.gov)," "NIH (nih.gov)"
                    ),
                    "SECONDARY_TRUSTED_SOURCES": (
                        "Google Scholar;" "Healthline (healthline.com)"
                    ),
                },
                clear=False,
            ):
                config = self._build_config(temp_dir)

            self.assertEqual(
                config.get_primary_trusted_sources(),
                ["PubMed (pubmed.ncbi.nlm.nih.gov)", "NIH (nih.gov)"],
            )
            self.assertEqual(
                config.get_secondary_trusted_sources(),
                ["Google Scholar", "Healthline (healthline.com)"],
            )
            self.assertEqual(
                config.trusted_sources,
                [
                    "PubMed (pubmed.ncbi.nlm.nih.gov)",
                    "NIH (nih.gov)",
                    "Google Scholar",
                    "Healthline (healthline.com)",
                ],
            )

    def test_ollama_model_is_preserved_in_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(
                temp_dir,
                llm_type=LLMType.OLLAMA,
                llm_model="llama3.1:8b",
            )

            self.assertEqual(config.llm_type, LLMType.OLLAMA)
            self.assertEqual(config.llm_model, "llama3.1:8b")

    def test_prompt_template_includes_source_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(temp_dir)
            prompt = config.get_prompt_template()

            self.assertIn("Source policy:", prompt)
            self.assertIn("Primary sources:", prompt)
            self.assertIn("Secondary sources:", prompt)
            self.assertIn("Prefer primary sources", prompt)
            self.assertIn("Google Scholar", prompt)


class WorkflowStateTests(unittest.TestCase):
    def test_state_round_trip_preserves_transitions(self):
        state = WorkflowState.from_raw(
            {
                "research_notes": {"Note 1": "Initial finding"},
                "draft_report_filename": "initial.md",
            },
            default_report_filename="fallback.md",
        )

        state.record_notes("Note 2", "Secondary finding")
        state.store_draft("Draft report", "draft.md")
        self.assertEqual(state.report_status, ReportStatus.DRAFT_READY)
        self.assertFalse(state.review_approved)

        state.request_changes("Add stronger citations")
        self.assertEqual(state.report_status, ReportStatus.CHANGES_REQUESTED)
        self.assertEqual(state.review_feedback, "Add stronger citations")

        state.approve()
        self.assertTrue(state.review_approved)
        self.assertEqual(state.report_status, ReportStatus.APPROVED)

        state.publish("docs/final.md")
        self.assertEqual(state.report_status, ReportStatus.PUBLISHED)
        self.assertEqual(state.final_report_path, "docs/final.md")

        raw_state = state.to_raw()
        self.assertEqual(raw_state["research_notes"]["Note 1"], "Initial finding")
        self.assertEqual(raw_state["research_notes"]["Note 2"], "Secondary finding")
        self.assertEqual(raw_state["report_status"], ReportStatus.PUBLISHED.value)

    def test_runtime_event_payload_is_ui_friendly(self):
        event = WorkflowRuntimeEvent(
            type=WorkflowEventType.COMPLETION,
            message="Workflow completed successfully",
        )

        payload = event.to_payload()

        self.assertEqual(payload["type"], "completion")
        self.assertEqual(payload["message"], "Workflow completed successfully")
        self.assertNotIn("agent", payload)


class WorkflowRuntimeStreamTests(unittest.TestCase):
    def _build_runtime_workflow(
        self,
        events=None,
        max_iterations: int = 3,
        timeout_seconds: float = 30,
        agent_workflow=None,
    ):
        workflow = object.__new__(MultiAgentWorkflow)
        workflow.config = FakeRuntimeConfig(
            max_iterations=max_iterations,
            timeout_seconds=timeout_seconds,
        )
        workflow.logger = logging.getLogger("workflow-runtime-test")
        workflow.agent_workflow = agent_workflow or FakeAgentWorkflow(events or [])
        return workflow

    def test_stream_runtime_events_normalizes_runtime_output(self):
        workflow = self._build_runtime_workflow(
            [
                FakeAgentChangeEvent("ResearchAgent"),
                FakeAgentOutputEvent(
                    content="Collected notes",
                    tool_calls=[FakeToolPlan("search_web")],
                ),
                FakeToolCallEvent("search_web", {"query": "health"}),
                FakeToolCallResultEvent(
                    "search_web",
                    {"query": "health"},
                    "result payload",
                ),
            ]
        )

        async def collect_events():
            return [event async for event in workflow.stream_runtime_events()]

        with patch("multi_agent_workflow.AgentOutput", FakeAgentOutputEvent), patch(
            "multi_agent_workflow.ToolCall", FakeToolCallEvent
        ), patch("multi_agent_workflow.ToolCallResult", FakeToolCallResultEvent):
            events = asyncio.run(collect_events())

        self.assertEqual(
            [event.type for event in events],
            [
                WorkflowEventType.AGENT_CHANGE,
                WorkflowEventType.AGENT_OUTPUT,
                WorkflowEventType.TOOL_CALL,
                WorkflowEventType.TOOL_RESULT,
                WorkflowEventType.COMPLETION,
            ],
        )
        self.assertEqual(events[0].agent, "ResearchAgent")
        self.assertEqual(events[1].tool_calls, ["search_web"])
        self.assertEqual(events[2].arguments, {"query": "health"})
        self.assertEqual(events[3].output, "result payload")
        self.assertEqual(events[4].message, "Workflow completed successfully")

    def test_stream_runtime_events_stops_before_exceeding_max_iterations(self):
        workflow = self._build_runtime_workflow(
            [
                FakeAgentChangeEvent("ResearchAgent"),
                FakeAgentOutputEvent(content="Collected notes"),
                FakeAgentChangeEvent("WriteAgent"),
            ],
            max_iterations=1,
        )

        async def collect_events():
            return [event async for event in workflow.stream_runtime_events()]

        with patch("multi_agent_workflow.AgentOutput", FakeAgentOutputEvent), patch(
            "multi_agent_workflow.ToolCall", FakeToolCallEvent
        ), patch("multi_agent_workflow.ToolCallResult", FakeToolCallResultEvent):
            events = asyncio.run(collect_events())

        self.assertEqual(
            [event.type for event in events],
            [
                WorkflowEventType.AGENT_CHANGE,
                WorkflowEventType.AGENT_OUTPUT,
                WorkflowEventType.COMPLETION,
            ],
        )
        self.assertIn("max iterations", events[-1].message)

    def test_stream_runtime_events_emits_timeout_when_no_event_arrives_in_time(self):
        workflow = self._build_runtime_workflow(
            timeout_seconds=0.01,
            agent_workflow=FakeSlowAgentWorkflow(0.05),
        )

        async def collect_events():
            return [event async for event in workflow.stream_runtime_events()]

        events = asyncio.run(collect_events())

        self.assertEqual(
            [event.type for event in events], [WorkflowEventType.TIMED_OUT]
        )
        self.assertIn("timed out", events[0].message)


class WorkflowDependencyIsolationTests(unittest.TestCase):
    def test_runtime_dependencies_are_stored_on_the_workflow_instance(self):
        workflow = object.__new__(MultiAgentWorkflow)
        workflow.config = type(
            "Config",
            (),
            {"llm_type": LLMType.OLLAMA, "llm_model": "llama3.1:8b"},
        )()
        workflow.logger = logging.getLogger("workflow-dependency-test")

        with patch(
            "multi_agent_workflow.get_embedding_model", return_value="embed"
        ), patch("multi_agent_workflow.get_llm", return_value="llm"):
            workflow._setup_runtime_dependencies()

        self.assertEqual(workflow.embedding_model, "embed")
        self.assertEqual(workflow.llm, "llm")

    def test_agents_receive_the_instance_llm(self):
        workflow = object.__new__(MultiAgentWorkflow)
        workflow.config = type("Config", (), {"target_word_count": 1500})()
        workflow.logger = logging.getLogger("workflow-dependency-test")
        workflow.llm = "instance-llm"
        workflow.search_web = "search-web"

        with patch("multi_agent_workflow.FunctionAgent") as function_agent_cls:
            function_agent_cls.side_effect = ["research", "write", "review"]
            workflow._setup_agents()

        self.assertEqual(function_agent_cls.call_count, 3)
        for call in function_agent_cls.call_args_list:
            self.assertEqual(call.kwargs["llm"], "instance-llm")


if __name__ == "__main__":
    unittest.main()
