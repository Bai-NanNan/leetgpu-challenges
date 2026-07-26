import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_challenges_to_md.py"
SPEC = importlib.util.spec_from_file_location("convert_challenges_to_md", SCRIPT)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)


def test_math_delimiters_survive_html_conversion(tmp_path):
    source = tmp_path / "challenge.html"
    destination = tmp_path / "challenge.md"
    source.write_text("<p>Inline \\(x_i\\).</p>\n\\[A = B \\\\ C\\]\n<p>End.</p>")

    converter.convert_html(source, destination)

    markdown = destination.read_text()
    assert "$x_i$" in markdown
    assert "$$\nA = B \\\\ C\n$$" in markdown
    assert "\\\\(x_i\\\\)" not in markdown
