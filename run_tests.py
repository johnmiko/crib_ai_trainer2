# Helper script to run pytest with the correct PYTHONPATH
import os
import sys
import subprocess

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = root
    sys.exit(subprocess.call([sys.executable, "-m", "pytest"] + sys.argv[1:], env=env))
