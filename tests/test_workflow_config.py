import tempfile
import unittest

from multi_agent_workflow import WorkflowConfig


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

    def test_prompt_template_includes_source_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(temp_dir)
            prompt = config.get_prompt_template()

            self.assertIn("Source policy:", prompt)
            self.assertIn("Primary sources:", prompt)
            self.assertIn("Secondary sources:", prompt)
            self.assertIn("Prefer primary sources", prompt)
            self.assertIn("Google Scholar", prompt)


if __name__ == "__main__":
    unittest.main()
