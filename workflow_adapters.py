"""Adapters that isolate external workflow dependencies from the engine."""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Protocol

from llama_index.core.tools import AsyncBaseTool
from llama_index.tools.tavily_research import TavilyToolSpec


class SearchToolAdapter(Protocol):
    """Provide search tools to the workflow engine."""

    def get_search_tools(self) -> list[AsyncBaseTool]:
        """Return the tools used for web search."""


class ReportPersistenceAdapter(Protocol):
    """Persist final report content without exposing file I/O to the engine."""

    def publish_report(self, report_content: str, filename: str) -> str:
        """Write the report and return the final filepath."""


class TavilySearchAdapter:
    """Adapter around the Tavily tool spec."""

    def __init__(self, api_key: str | None):
        self.api_key = api_key

    def get_search_tools(self) -> list[AsyncBaseTool]:
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")

        tavily_tool = TavilyToolSpec(api_key=self.api_key)
        tavily_tools = tavily_tool.to_tool_list()

        if not tavily_tools:
            raise ValueError("No Tavily tools available")

        return tavily_tools


class LocalReportPersistenceAdapter:
    """Persist reports to the configured docs directory."""

    def __init__(self, docs_dir: str, default_report_filename: str):
        self.docs_dir = docs_dir
        self.default_report_filename = default_report_filename

    def _build_filepath(self, filename: str | None = None) -> str:
        target_filename = filename or self.default_report_filename
        base, ext = os.path.splitext(target_filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_with_stamp = f"{base}_{timestamp}{ext}"
        return os.path.join(self.docs_dir, filename_with_stamp)

    def publish_report(self, report_content: str, filename: str) -> str:
        filepath = self._build_filepath(filename)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as file_handle:
            file_handle.write(report_content)

        return filepath
