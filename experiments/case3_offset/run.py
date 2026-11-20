"""Run case3_offset with the frozen experiment driver."""
from pathlib import Path
import subprocess


def main():
    subprocess.run(["bash", str(Path(__file__).with_name("run.sh"))], check=True)


if __name__ == "__main__":
    main()
