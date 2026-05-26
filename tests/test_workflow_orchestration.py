import logging
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
import unittest

from multi_agent_workflow import (
    LLMType,
    MultiAgentWorkflow,
    ReportStatus,
    WorkflowState,
)


class FakeContext:
    def __init__(self, initial_state):
        self._state = initial_state

    async def get(self, key, default=None):
        if key == "state":
            return self._state
        return default

    async def set(self, key, value):
        if key == "state":
            self._state = value


class FakeSearchAdapter:
    def __init__(self):
        self.calls = 0

    def get_search_tools(self):
        self.calls += 1
        return ["search-web-tool"]


class FakePersistenceAdapter:
    def __init__(self):
        self.calls = []

    def publish_report(self, report_content: str, filename: str) -> str:
        filepath = str(PurePosixPath("/virtual-docs") / filename)
        self.calls.append(
            {
                "filepath": filepath,
                "content": report_content,
                "filename": filename,
            }
        )
        return filepath


def build_workflow(fake_search_adapter=None, fake_persistence_adapter=None):
    workflow = object.__new__(MultiAgentWorkflow)
    workflow.config = SimpleNamespace(
        tavily_api_key="test-key",
        llm_type=LLMType.OLLAMA,
        llm_model="llama3.1:8b",
        docs_dir="/virtual-docs",
        default_report_filename="report.md",
        target_word_count=1500,
        min_developments=1,
        max_developments=2,
        max_iterations=10,
        timeout_seconds=30,
    )
    workflow.logger = logging.getLogger("workflow-orchestration-test")
    workflow.search_adapter = fake_search_adapter or FakeSearchAdapter()
    workflow.persistence_adapter = fake_persistence_adapter or FakePersistenceAdapter()
    workflow.search_web = "search-web-tool"
    return workflow


class WorkflowOrchestrationIntegrationTests(IsolatedAsyncioTestCase):
    async def test_research_draft_approve_publish_flow(self):
        fake_search_adapter = FakeSearchAdapter()
        fake_persistence_adapter = FakePersistenceAdapter()
        workflow = build_workflow(fake_search_adapter, fake_persistence_adapter)

        workflow._setup_tools()
        self.assertEqual(fake_search_adapter.calls, 1)
        self.assertEqual(workflow.search_web, "search-web-tool")

        ctx = FakeContext(workflow._create_initial_state())
        record_notes = workflow._create_record_notes_tool()
        write_report = workflow._create_write_report_tool()
        approve_report = workflow._create_approve_report_tool()
        publish_report = workflow._create_publish_report_tool()

        notes_result = await record_notes(
            ctx,
            notes="PubMed and NIH sources support the claim.",
            notes_title="Research note 1",
        )
        draft_result = await write_report(
            ctx,
            report_content="# Draft\n\n## Research note 1\nPubMed and NIH sources support the claim.",
            filename="health-report.md",
        )
        approval_result = await approve_report(ctx)
        publish_result = await publish_report(ctx)

        final_state = WorkflowState.from_raw(
            await ctx.get("state"), default_report_filename="report.md"
        )

        self.assertEqual(notes_result, "Notes recorded successfully.")
        self.assertEqual(
            draft_result, "Draft report stored. Awaiting review before publishing."
        )
        self.assertEqual(approval_result, "Report approved successfully.")
        self.assertIn("Final report published to", publish_result)

        self.assertEqual(final_state.report_status, ReportStatus.PUBLISHED)
        self.assertTrue(final_state.review_approved)
        self.assertEqual(
            final_state.research_notes["Research note 1"],
            "PubMed and NIH sources support the claim.",
        )
        self.assertEqual(final_state.draft_report_filename, "health-report.md")
        self.assertEqual(
            final_state.final_report_path, "/virtual-docs/health-report.md"
        )
        self.assertEqual(len(fake_persistence_adapter.calls), 1)
        self.assertEqual(
            fake_persistence_adapter.calls[0]["content"],
            "# Draft\n\n## Research note 1\nPubMed and NIH sources support the claim.",
        )


if __name__ == "__main__":
    unittest.main()
