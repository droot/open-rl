"""The skill's settings table is generated from class Settings; this fails when
someone changes a setting without rerunning scripts/gen_settings_doc.py."""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_generator():
  spec = importlib.util.spec_from_file_location("gen_settings_doc", ROOT / "scripts" / "gen_settings_doc.py")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class SkillDocsTest(unittest.TestCase):
  def test_settings_table_matches_the_settings_model(self):
    gen = load_generator()
    checked_in = gen.TARGET.read_text()
    self.assertEqual(checked_in, gen.render(), "skills/openrl/references/settings.md is stale; run scripts/gen_settings_doc.py")

  def test_every_setting_has_a_row(self):
    from server.api_server import Settings

    table = load_generator().TARGET.read_text()
    for name in Settings.model_fields:
      self.assertIn(f"`openrl.{name}`", table)

  def test_skill_frontmatter_names_the_skill(self):
    text = (ROOT / "skills" / "openrl" / "SKILL.md").read_text()
    self.assertTrue(text.startswith("---\nname: openrl\n"), "SKILL.md needs the name: openrl frontmatter")
    self.assertIn("description:", text.split("---")[1])


if __name__ == "__main__":
  unittest.main()
