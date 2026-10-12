"""Render the openrl.* settings table for the skill from the server's own catalog.

  uv run --extra cpu python scripts/gen_settings_doc.py          # rewrite skills/openrl/references/settings.md
  uv run --extra cpu python scripts/gen_settings_doc.py --check  # exit 1 if the file is stale

tests/test_skill_docs.py runs the check, so the table cannot drift from
class Settings in src/server/api_server.py (design 016, section 5.5).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from server.api_server import settings_catalog  # noqa: E402

TARGET = ROOT / "skills" / "openrl" / "references" / "settings.md"

HEADER = """# openrl.* settings

Generated from `class Settings` in `src/server/api_server.py` by
`scripts/gen_settings_doc.py`; do not edit by hand. A running server reports
the same table under `openrl.settings` in `GET /api/v1/get_server_capabilities`.

Pass settings as `user_metadata` on `create_session` or `create_model`, or as
`TINKER_TAGS=openrl.key=value,openrl.other=value` in the client's environment.
Model settings override session settings. Values are strings on the wire
(`"true"`, `"8"`); an unknown `openrl.` key or a bad value is a 400 that names
the setting.

```python
client = tinker.ServiceClient(base_url=..., user_metadata={"openrl.fine_tuning_type": "full"})
```

"""


def fmt(value) -> str:
  if value is None:
    return "unset"
  if isinstance(value, bool):
    return str(value).lower()
  return f"`{value}`"


def render() -> str:
  rows = ["| Setting | Type | Default | Values | Description |", "|---|---|---|---|---|"]
  for entry in settings_catalog():
    values = ", ".join(f"`{v}`" for v in entry["allowed"]) if "allowed" in entry else ""
    if "min" in entry or "max" in entry:
      values = f"{entry.get('min', '')} to {entry.get('max', '')}"
    rows.append(f"| `{entry['name']}` | {entry['type']} | {fmt(entry['default'])} | {values} | {entry['description']} |")
  return HEADER + "\n".join(rows) + "\n"


def main() -> None:
  text = render()
  if "--check" in sys.argv:
    current = TARGET.read_text() if TARGET.exists() else ""
    if current != text:
      print(f"{TARGET.relative_to(ROOT)} is stale; run scripts/gen_settings_doc.py", file=sys.stderr)
      sys.exit(1)
    print("settings.md is current")
    return
  TARGET.parent.mkdir(parents=True, exist_ok=True)
  TARGET.write_text(text)
  print(f"wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
  main()
