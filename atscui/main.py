import os
import sys
from pathlib import Path
from atscui.ui.main import ATSCUI

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

def setup_environment():
    """Setup necessary environment variables and paths"""
    if "SUMO_HOME" not in os.environ:
        sys.exit("Please declare the environment variable 'SUMO_HOME'")

    # Add SUMO tools to Python path
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.append(str(project_root))


def main():
    setup_environment()

    ui = ATSCUI()
    demo = ui.create_ui()
    demo.launch()


if __name__ == "__main__":
    main()