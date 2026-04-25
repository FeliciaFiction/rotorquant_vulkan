from pathlib import Path


def shader_dir() -> Path:
    return Path(__file__).resolve().parent / "shaders"

