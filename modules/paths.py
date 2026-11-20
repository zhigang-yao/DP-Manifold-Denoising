"""Artifact paths independent of the caller's working directory."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
OUTPUTS=ROOT/"outputs"
