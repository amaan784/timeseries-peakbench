"""The generated results only ever replace the text between their two markers."""

from src import report


def test_fill_markers_replaces_only_between_the_markers(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        f"# Title\n\nhand-written intro\n\n{report.RESULTS_START}\nold numbers\n{report.RESULTS_END}\n\n"
        "hand-written analysis\n",
        encoding="utf-8",
    )
    assert report.fill_markers(page, "new numbers")
    text = page.read_text(encoding="utf-8")
    assert "new numbers" in text
    assert "old numbers" not in text
    assert text.startswith("# Title\n\nhand-written intro\n\n")
    assert text.endswith("hand-written analysis\n")

    # running it again changes nothing
    report.fill_markers(page, "new numbers")
    assert page.read_text(encoding="utf-8") == text


def test_files_without_markers_are_left_alone(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("no markers here\n", encoding="utf-8")
    assert not report.fill_markers(page, "numbers")
    assert page.read_text(encoding="utf-8") == "no markers here\n"
    assert not report.fill_markers(tmp_path / "missing.md", "numbers")
