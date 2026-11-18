"""Run UniT reconstruction and save predictions without a viewer."""

from pathlib import Path

from unit.inference.cli import main

if __name__ == "__main__":
    main(project_root=Path(__file__).resolve().parent)
