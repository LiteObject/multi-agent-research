"""Research profile loading and prompt composition helpers.

Research profiles keep topic-specific settings outside the workflow engine.
Environment variables can override the active profile path and the source tiers,
while the workflow code only consumes the resulting profile object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml

DEFAULT_PRIMARY_SOURCES = [
    "PubMed (pubmed.ncbi.nlm.nih.gov)",
    "The Lancet (thelancet.com)",
    "Nature Medicine (nature.com/nm)",
    "ScienceDirect (sciencedirect.com)",
    "National Institutes of Health (NIH) (nih.gov)",
    "World Health Organization (WHO) (who.int)",
    "Centers for Disease Control and Prevention (CDC) (cdc.gov)",
    "MedlinePlus (medlineplus.gov)",
]

DEFAULT_SECONDARY_SOURCES = [
    "Google Scholar",
    "ScienceDaily (sciencedaily.com)",
    "Medical News Today (medicalnewstoday.com)",
    "Healthline (healthline.com)",
    "Everyday Health (everydayhealth.com)",
    "Harvard Health Blog (health.harvard.edu)",
    "Kaiser Health News (KHN) (kffhealthnews.org)",
    "WebMD Doctors Blog (webmd.com)",
    "MobiHealthNews (mobihealthnews.com)",
    "HIMSS (Healthcare Information and Management Systems Society) (himss.org)",
    "American Public Health Association (APHA) (apha.org)",
    "TEDMED (tedmed.com)",
]


def parse_source_list(raw_sources: Optional[str]) -> List[str]:
    """Parse a comma, semicolon, or newline separated source list."""
    if not raw_sources:
        return []

    normalized = raw_sources.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", ",").replace(";", ",")
    return [source.strip() for source in normalized.split(",") if source.strip()]


@dataclass
class ResearchProfile:
    """Topic-specific settings used to generate research prompts."""

    name: str
    title: str
    topic: str
    audience: str = "general readers"
    tone: str = "informative yet conversational"
    impact_focus: str = "everyday health"
    subject_background: str = "the subject matter"
    target_word_count: int = 5000
    min_developments: int = 5
    max_developments: int = 7
    default_report_filename: str = "report.md"
    primary_sources: List[str] = field(default_factory=list)
    secondary_sources: List[str] = field(default_factory=list)
    report_rules: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchProfile":
        """Build a profile from YAML or JSON-style data."""
        return cls(
            name=str(data.get("name", "custom_research")),
            title=str(data.get("title", "Multi-Agent Research")),
            topic=str(data.get("topic", "research")),
            audience=str(data.get("audience", "general readers")),
            tone=str(data.get("tone", "informative yet conversational")),
            impact_focus=str(data.get("impact_focus", "everyday health")),
            subject_background=str(
                data.get("subject_background", "the subject matter")
            ),
            target_word_count=int(data.get("target_word_count", 5000)),
            min_developments=int(data.get("min_developments", 5)),
            max_developments=int(data.get("max_developments", 7)),
            default_report_filename=str(
                data.get("default_report_filename", "report.md")
            ),
            primary_sources=[str(item) for item in data.get("primary_sources", [])],
            secondary_sources=[str(item) for item in data.get("secondary_sources", [])],
            report_rules=[str(item) for item in data.get("report_rules", [])],
        )

    def build_source_guidance(
        self,
        primary_sources: Optional[List[str]] = None,
        secondary_sources: Optional[List[str]] = None,
    ) -> str:
        """Return the source policy block used by prompts."""
        primary = (
            primary_sources if primary_sources is not None else self.primary_sources
        )
        secondary = (
            secondary_sources
            if secondary_sources is not None
            else self.secondary_sources
        )

        primary_block = (
            "\n".join(f"- {source}" for source in primary)
            if primary
            else "- None configured"
        )
        secondary_block = (
            "\n".join(f"- {source}" for source in secondary)
            if secondary
            else "- None configured"
        )

        return (
            "Source policy:\n"
            "Primary sources should be used first for claims, statistics, outcomes, approvals, and dates.\n"
            "Secondary sources should be used for discovery or context only, and any factual claim from them should be corroborated by a primary source.\n\n"
            f"Primary sources:\n{primary_block}\n\n"
            f"Secondary sources:\n{secondary_block}\n\n"
            "Rules:\n"
            "- Prefer primary sources for the final citation whenever possible.\n"
            "- Do not treat Google Scholar as a final citation; use it to locate the underlying paper.\n"
            "- If a secondary source is the only lead found, verify it against a primary source before using it in the report."
        )

    def build_prompt(
        self,
        *,
        target_word_count: int,
        min_developments: int,
        max_developments: int,
        current_month_year: str,
        source_guidance: str,
    ) -> str:
        """Return the full prompt for the active research profile."""
        extra_rules = (
            "\n".join(f"- {rule}" for rule in self.report_rules)
            if self.report_rules
            else "- Follow the source policy below."
        )

        return f"""
        Write a {target_word_count}-word blog post that highlights and explains
        {min_developments} to {max_developments} of the most recent and significant
        developments in {self.topic} as of {current_month_year}. Use the most up-to-date
        information using the source policy below. For each development, cite your sources in-line in markdown format, including the article or study title,
        author(s) if available, publication date, and a direct URL. Focus on topics that
        are relevant to {self.audience} and provide clear, accessible explanations of each
        breakthrough. Include relevant statistics, cite the source and publication date of
        each study or article, and end with a takeaway section summarizing why these updates
        matter for {self.impact_focus}. Maintain a tone that is {self.tone},
        suitable for readers who may not have a background in {self.subject_background}.

        {extra_rules}

        {source_guidance}
        """


@dataclass(frozen=True)
class ResearchProfileSelection:
    """A selectable research profile entry for the UI."""

    label: str
    path: Optional[str]
    profile: ResearchProfile


def _resolve_profile_path(profile_path: Optional[str]) -> Optional[Path]:
    if not profile_path:
        return None

    candidate = Path(profile_path)
    if candidate.is_absolute():
        return candidate

    return Path(__file__).resolve().parent / candidate


def _resolve_profile_directory(profile_directory: Optional[str] = None) -> Path:
    if not profile_directory:
        return Path(__file__).resolve().parent / "research_profiles"

    candidate = Path(profile_directory)
    if candidate.is_absolute():
        return candidate

    return Path(__file__).resolve().parent / candidate


def discover_research_profile_paths(
    profile_directory: Optional[str] = None,
) -> List[Path]:
    """Return YAML profile files available to the picker."""
    directory = _resolve_profile_directory(profile_directory)
    if not directory.exists():
        return []

    candidates = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))
    unique_candidates = {candidate.resolve(): candidate for candidate in candidates}
    return sorted(
        unique_candidates.values(), key=lambda candidate: candidate.stem.lower()
    )


def get_research_profile_options(
    profile_directory: Optional[str] = None,
) -> List[ResearchProfileSelection]:
    """Return selectable profile entries for the sidebar."""
    selections = [
        ResearchProfileSelection(
            label=f"{build_default_research_profile().title} (built-in default)",
            path=None,
            profile=build_default_research_profile(),
        )
    ]

    for profile_path in discover_research_profile_paths(profile_directory):
        try:
            profile = load_research_profile(str(profile_path))
        except (FileNotFoundError, ValueError):
            continue

        selections.append(
            ResearchProfileSelection(
                label=f"{profile.title} ({profile_path.stem})",
                path=str(profile_path),
                profile=profile,
            )
        )

    return selections


def build_default_research_profile() -> ResearchProfile:
    """Return the built-in health science profile used as a safe fallback."""
    return ResearchProfile(
        name="health_science",
        title="Health Science Research Agent",
        topic="health science",
        audience="general readers",
        tone="informative yet conversational",
        impact_focus="everyday health",
        subject_background="medical topics",
        target_word_count=5000,
        min_developments=5,
        max_developments=7,
        default_report_filename="health_report.md",
        primary_sources=list(DEFAULT_PRIMARY_SOURCES),
        secondary_sources=list(DEFAULT_SECONDARY_SOURCES),
        report_rules=[
            "Prefer primary sources for the final citation whenever possible.",
            "Do not treat Google Scholar as a final citation; use it to locate the underlying paper.",
            "If a secondary source is the only lead found, verify it against a primary source before using it in the report.",
        ],
    )


def load_research_profile(profile_path: Optional[str] = None) -> ResearchProfile:
    """Load a research profile from YAML, or return the built-in fallback."""
    resolved_path = _resolve_profile_path(profile_path)
    if resolved_path is None:
        return build_default_research_profile()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Research profile not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(
            f"Research profile must contain a mapping at {resolved_path}, got {type(payload).__name__}"
        )

    return ResearchProfile.from_dict(payload)


def get_effective_research_profile(
    profile_path: Optional[str] = None,
) -> ResearchProfile:
    """Load the active profile and apply source overrides from the environment."""
    profile = load_research_profile(profile_path)

    primary_override = parse_source_list(os.getenv("PRIMARY_TRUSTED_SOURCES"))
    secondary_override = parse_source_list(os.getenv("SECONDARY_TRUSTED_SOURCES"))

    if primary_override:
        profile.primary_sources = primary_override
    if secondary_override:
        profile.secondary_sources = secondary_override

    return profile
