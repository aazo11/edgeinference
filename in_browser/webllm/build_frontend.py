#!/usr/bin/env python3
"""
Utility script to build the frontend for the WebLLM benchmark.
This script will build the frontend and place it in the dist directory.
"""

import os
import subprocess
import sys
import shutil

def build_frontend():
    """Build the frontend for the WebLLM benchmark."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(current_dir, 'frontend')
    dist_dir = os.path.join(frontend_dir, 'dist')
    
    # Check if we have webpack and npm
    npm_path = shutil.which('npm')
    if npm_path is None:
        print("Error: npm not found. Please install Node.js and npm.")
        return False
    
    # Check if the frontend directory exists
    if not os.path.exists(frontend_dir):
        print(f"Error: Frontend directory {frontend_dir} not found.")
        return False
    
    # Always install dependencies to ensure they're up to date
    print("Installing frontend dependencies...")
    try:
        subprocess.run(
            ['npm', 'install'],
            cwd=frontend_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print(f"stdout: {e.stdout.decode('utf-8')}")
        print(f"stderr: {e.stderr.decode('utf-8')}")
        return False
    
    # Build the frontend
    print("Building frontend...")
    try:
        subprocess.run(
            ['npm', 'run', 'build'],
            cwd=frontend_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"Error building frontend: {e}")
        print(f"stdout: {e.stdout.decode('utf-8')}")
        print(f"stderr: {e.stderr.decode('utf-8')}")
        return False
    
    # Check if the build was successful
    if not os.path.exists(dist_dir):
        print(f"Error: Build directory {dist_dir} not found after build.")
        return False
    
    dist_index = os.path.join(dist_dir, 'index.html')
    if not os.path.exists(dist_index):
        print(f"Error: Built index.html not found at {dist_index}")
        return False
    
    print(f"Frontend built successfully at {dist_dir}")
    return True

if __name__ == "__main__":
    success = build_frontend()
    sys.exit(0 if success else 1) 