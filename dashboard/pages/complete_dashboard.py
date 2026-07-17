from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]

runpy.run_path(
    str(PROJECT_ROOT / "app_single_page_backup.py"),
    run_name="__main__",
)
