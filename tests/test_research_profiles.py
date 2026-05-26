import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from research_profiles import (
    ResearchProfile,
    get_research_profile_options,
    load_research_profile,
    parse_source_list,
)
from multi_agent_workflow import WorkflowConfig


class ResearchProfileTests(unittest.TestCase):
    def test_parse_source_list_supports_newlines(self):
        sources = parse_source_list("PubMed\nThe Lancet;Nature Medicine,ScienceDirect")

        self.assertEqual(
            sources,
            ["PubMed", "The Lancet", "Nature Medicine", "ScienceDirect"],
        )

        def test_profile_picker_discovers_builtin_and_yaml_profiles(self):
            profile_dir = Path(__file__).resolve().parents[1] / "research_profiles"

            options = get_research_profile_options(str(profile_dir))
            labels = [option.label for option in options]

            self.assertIn("Health Science Research Agent (built-in default)", labels)
            self.assertIn("Health Science Research Agent (health_science)", labels)

            yaml_option = next(option for option in options if option.path is not None)
            self.assertEqual(Path(yaml_option.path).name, "health_science.yaml")

        def test_load_research_profile_from_yaml(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                profile_path = Path(temp_dir) / "profile.yaml"
                profile_path.write_text(
                    textwrap.dedent("""
                    name: climate_research
                    title: Climate Research Agent
                    topic: climate policy
                    audience: policy analysts
                    tone: concise and analytical
                    impact_focus: public planning
                    subject_background: policy analysis
                    target_word_count: 3200
                    min_developments: 3
                    max_developments: 5
                    default_report_filename: climate_report.md
                    primary_sources:
                      - IPCC
                      - NOAA
                    secondary_sources:
                      - Reuters
                    report_rules:
                      - Lead with the strongest primary source.
                    """).strip(),
                    encoding="utf-8",
                )

                profile = load_research_profile(str(profile_path))

                self.assertIsInstance(profile, ResearchProfile)
                self.assertEqual(profile.name, "climate_research")
                self.assertEqual(profile.title, "Climate Research Agent")
                self.assertEqual(profile.topic, "climate policy")
                self.assertEqual(profile.target_word_count, 3200)
                self.assertEqual(profile.primary_sources, ["IPCC", "NOAA"])
                self.assertEqual(profile.secondary_sources, ["Reuters"])
                self.assertEqual(
                    profile.report_rules, ["Lead with the strongest primary source."]
                )

        def test_workflow_config_uses_explicit_profile_path(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                profile_path = Path(temp_dir) / "profile.yaml"
                profile_path.write_text(
                    textwrap.dedent("""
                        name: climate_research
                        title: Climate Research Agent
                        topic: climate policy
                        audience: policy analysts
                        tone: concise and analytical
                        impact_focus: public planning
                        subject_background: policy analysis
                        target_word_count: 3200
                        min_developments: 3
                        max_developments: 5
                        default_report_filename: climate_report.md
                        primary_sources:
                          - IPCC
                          - NOAA
                        secondary_sources:
                          - Reuters
                        report_rules:
                          - Lead with the strongest primary source.
                        """.strip()),
                    encoding="utf-8",
                )

                config = WorkflowConfig(
                    tavily_api_key="test-key",
                    research_profile_path=str(profile_path),
                    docs_dir=temp_dir,
                    max_iterations=3,
                    timeout_seconds=60,
                )

                self.assertEqual(config.research_profile.name, "climate_research")
                self.assertEqual(config.target_word_count, 3200)
                self.assertEqual(config.min_developments, 3)
                self.assertEqual(config.max_developments, 5)
                self.assertEqual(config.default_report_filename, "climate_report.md")
                self.assertEqual(config.get_primary_trusted_sources(), ["IPCC", "NOAA"])
                self.assertEqual(config.get_secondary_trusted_sources(), ["Reuters"])

                prompt = config.get_prompt_template()
                self.assertIn("climate policy", prompt)
                self.assertIn("policy analysts", prompt)
                self.assertIn("concise and analytical", prompt)
                self.assertIn("public planning", prompt)

        def test_workflow_config_uses_profile_defaults(self):
            with tempfile.TemporaryDirectory() as temp_dir:
                profile_path = Path(temp_dir) / "profile.yaml"
                profile_path.write_text(
                    textwrap.dedent("""
                    name: climate_research
                    title: Climate Research Agent
                    topic: climate policy
                    audience: policy analysts
                    tone: concise and analytical
                    impact_focus: public planning
                    subject_background: policy analysis
                    target_word_count: 3200
                    min_developments: 3
                    max_developments: 5
                    default_report_filename: climate_report.md
                    primary_sources:
                      - IPCC
                      - NOAA
                    secondary_sources:
                      - Reuters
                    report_rules:
                      - Lead with the strongest primary source.
                    """).strip(),
                    encoding="utf-8",
                )

                with patch.dict(
                    os.environ,
                    {
                        "TAVILY_API_KEY": "test-key",
                        "RESEARCH_PROFILE_PATH": str(profile_path),
                    },
                    clear=False,
                ):
                    config = WorkflowConfig(
                        tavily_api_key="test-key",
                        docs_dir=temp_dir,
                        max_iterations=3,
                        timeout_seconds=60,
                    )

                self.assertEqual(config.target_word_count, 3200)
                self.assertEqual(config.min_developments, 3)
                self.assertEqual(config.max_developments, 5)
                self.assertEqual(config.default_report_filename, "climate_report.md")
                self.assertEqual(config.get_primary_trusted_sources(), ["IPCC", "NOAA"])
                self.assertEqual(config.get_secondary_trusted_sources(), ["Reuters"])

                prompt = config.get_prompt_template()
                self.assertIn("climate policy", prompt)
                self.assertIn("policy analysts", prompt)
                self.assertIn("concise and analytical", prompt)
                self.assertIn("public planning", prompt)


if __name__ == "__main__":
    unittest.main()
