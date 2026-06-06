from pathlib import Path
import tomllib

with open("pyproject.toml", "rb") as f:
    version = tomllib.load(f)["tool"]["poetry"]["version"]

Path("/version.py").write_text(
    f'__version__ = "{version}"\n'
)
