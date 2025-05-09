import argparse
import logging
import os
import sys

def main():
    """Main entry point for dashboard CLI"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="YouTube Network Dashboard")
    parser.add_argument("--data", type=str, help="Path to video data file")
    parser.add_argument("--analysis", type=str, help="Path to analysis results file")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the dashboard on")
    
    args = parser.parse_args()
    
    # Set environment variables for Streamlit
    if args.data:
        os.environ["ANCIENT_SQUIRREL_DATA"] = args.data
    if args.analysis:
        os.environ["ANCIENT_SQUIRREL_ANALYSIS"] = args.analysis
    
    # Run the Streamlit app
    import streamlit.web.cli as stcli
    
    # Get the path to the dashboard app
    from pathlib import Path
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard app not found at {dashboard_path}")
        return 1
    
    # Construct Streamlit arguments
    sys.argv = [
        "streamlit", "run", 
        str(dashboard_path),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    # Run Streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()