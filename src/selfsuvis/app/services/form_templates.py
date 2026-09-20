"""Load and serve HTML form templates."""

from pathlib import Path


def get_index_form_html() -> str:
    """Load and return the index video form HTML template."""
    template_dir = Path(__file__).resolve().parent.parent / "templates"
    template_path = template_dir / "index_form.html"
    return template_path.read_text(encoding="utf-8")
