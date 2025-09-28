
#!/usr/bin/env python3
"""
Main script to run the Financial Signal Generation System
"""
import streamlit.web.cli as stcli
import sys
from pathlib import Path

def main():
    """Run the Streamlit application"""

    # Add the current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

    # Run Streamlit
    streamlit_file = current_dir / "streamlit_app.py"

    sys.argv = [
        "streamlit",
        "run",
        str(streamlit_file),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=true"
    ]

    stcli.main()

if __name__ == "__main__":
    main()
