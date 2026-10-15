import os

from src import config


def test_paths_point_inside_the_project():
    assert (config.ROOT / "src" / "config.py").exists()
    assert config.CACHE.parent == config.OUT


def test_dotenv_reads_keys_without_overriding_the_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# a comment\nFOO_TEST_KEY='abc'\nALREADY_SET=from-file\nnot a setting\n")
    monkeypatch.delenv("FOO_TEST_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from-env")
    config.load_dotenv(env)
    assert os.environ["FOO_TEST_KEY"] == "abc"
    assert os.environ["ALREADY_SET"] == "from-env"
    monkeypatch.delenv("FOO_TEST_KEY")


def test_missing_dotenv_is_fine(tmp_path):
    config.load_dotenv(tmp_path / "nope.env")
