"""Tests for the DVC pipeline scripts.

These scripts were previously untested, which is why two real bugs survived:
extract_bronze.py never copied any files, and preprocess_silver.py used a
non-recursive glob so it found nothing once the bronze layer preserved class
subdirectories.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACT = REPO_ROOT / "src" / "scripts" / "extract_bronze.py"
PREPROCESS = REPO_ROOT / "src" / "scripts" / "preprocess_silver.py"


def _make_image(path: Path, size=(32, 32), color="white"):
    """Write a small real JPEG so PIL can open it."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG")


def _run(script: Path, *args):
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


class TestExtractBronze:
    def test_copies_files_and_preserves_class_folders(self, tmp_path):
        src = tmp_path / "raw"
        _make_image(src / "fertile" / "a.jpg")
        _make_image(src / "infertile" / "a.jpg")
        out = tmp_path / "bronze"

        result = _run(EXTRACT, "--source", str(src), "--output", str(out))

        assert result.returncode == 0, result.stderr
        # Same filename in two class folders must not collide.
        assert (out / "fertile" / "a.jpg").exists()
        assert (out / "infertile" / "a.jpg").exists()

    def test_reports_the_number_copied(self, tmp_path):
        src = tmp_path / "raw"
        _make_image(src / "fertile" / "a.jpg")
        _make_image(src / "fertile" / "b.jpg")
        out = tmp_path / "bronze"

        result = _run(EXTRACT, "--source", str(src), "--output", str(out))

        assert "2 files copied" in result.stdout

    def test_missing_source_raises(self, tmp_path):
        result = _run(
            EXTRACT, "--source", str(tmp_path / "nope"), "--output", str(tmp_path / "out")
        )
        assert result.returncode != 0
        assert "Source directory not found" in result.stderr


class TestPreprocessSilver:
    def test_finds_images_in_class_subdirectories(self, tmp_path):
        """Regression: a non-recursive glob found zero images in nested folders."""
        bronze = tmp_path / "bronze"
        _make_image(bronze / "fertile" / "a.jpg", size=(64, 64))
        _make_image(bronze / "infertile" / "b.jpg", size=(64, 64))
        silver = tmp_path / "silver"

        result = _run(
            PREPROCESS,
            "--input",
            str(bronze),
            "--output",
            str(silver),
            "--img-size",
            "32",
        )

        assert result.returncode == 0, result.stderr
        assert "2 images preprocessed" in result.stdout
        assert (silver / "fertile" / "a.jpg").exists()
        assert (silver / "infertile" / "b.jpg").exists()

    def test_resizes_to_the_requested_size(self, tmp_path):
        from PIL import Image

        bronze = tmp_path / "bronze"
        _make_image(bronze / "fertile" / "a.jpg", size=(128, 128))
        silver = tmp_path / "silver"

        _run(
            PREPROCESS, "--input", str(bronze), "--output", str(silver), "--img-size", "48"
        )

        with Image.open(silver / "fertile" / "a.jpg") as img:
            assert img.size == (48, 48)

    def test_reports_failures_instead_of_hiding_them(self, tmp_path):
        bronze = tmp_path / "bronze"
        bronze.mkdir(parents=True)
        # A file with an image extension but invalid contents.
        (bronze / "broken.jpg").write_bytes(b"not an image")
        silver = tmp_path / "silver"

        result = _run(PREPROCESS, "--input", str(bronze), "--output", str(silver))

        assert result.returncode == 0
        assert "1 failed" in result.stdout


class TestDvcPipelineDefinition:
    """The dvc.yaml must reference files that exist."""

    @pytest.fixture(scope="class")
    @classmethod
    def dvc_yaml(cls):
        yaml = pytest.importorskip("yaml")
        path = REPO_ROOT / "dvc.yaml"
        if not path.exists():
            pytest.skip("dvc.yaml not present")
        return yaml.safe_load(path.read_text())

    def test_every_stage_references_an_existing_script(self, dvc_yaml):
        for name, stage in dvc_yaml["stages"].items():
            cmd = stage["cmd"]
            for token in cmd.split():
                if token.endswith(".py"):
                    assert (REPO_ROOT / token).exists(), f"{name} references missing {token}"

    def test_every_dependency_exists(self, dvc_yaml):
        for name, stage in dvc_yaml["stages"].items():
            for dep in stage.get("deps", []):
                # Data paths are produced by earlier stages and may be absent
                # on a fresh clone; only source files must exist.
                if dep.endswith(".py"):
                    assert (REPO_ROOT / dep).exists(), f"{name} depends on missing {dep}"

    def test_declared_params_exist_in_params_yaml(self, dvc_yaml):
        yaml = pytest.importorskip("yaml")
        params_path = REPO_ROOT / "params.yaml"
        if not params_path.exists():
            pytest.skip("params.yaml not present")
        params = yaml.safe_load(params_path.read_text())

        for name, stage in dvc_yaml["stages"].items():
            for param in stage.get("params", []):
                section, _, key = param.partition(".")
                assert section in params, f"{name} references unknown param section {section}"
                assert key in params[section], f"{name} references unknown param {param}"
