from pathlib import Path


def test_readme_has_live_demo_sections():
    readme = Path("README.md").read_text(encoding="utf-8")
    for section in ["## Run live demo", "## Deploy for free", "## What makes this end-to-end"]:
        assert section in readme
