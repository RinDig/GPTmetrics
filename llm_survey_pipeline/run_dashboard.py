#!/usr/bin/env python
"""Launch script for the LLM Survey Pipeline Dashboard"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    # Ensure we're in the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Add parent directory to Python path
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    os.chdir(script_dir)
    
    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env['PYTHONPATH'] = parent_dir + os.pathsep + env.get('PYTHONPATH', '')
    
    # Launch streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"], env=env)

if __name__ == "__main__":
    main()