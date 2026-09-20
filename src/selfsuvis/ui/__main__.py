import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app = str(Path(__file__).parent / "app.py")
    sys.exit(subprocess.call(["streamlit", "run", app, *sys.argv[1:]]))
