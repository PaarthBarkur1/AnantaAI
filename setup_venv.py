#!/usr/bin/env python3
"""
Setup script to create and configure a virtual environment for AnantaAI.
This ensures all dependencies are properly installed in an isolated environment.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}...")
    try:
        result = subprocess.run(command, check=True,
                                capture_output=True, text=True)
        print(f"SUCCESS: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {description} failed:")
        print(f"   Command: {' '.join(command)}")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Set up virtual environment for AnantaAI."""
    print("AnantaAI Virtual Environment Setup")
    print("=" * 50)

    # Check if we're already in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("WARNING: You're already in a virtual environment!")
        print("   Please deactivate it first and run this script again.")
        return 1

    # Get the project directory
    project_dir = Path(__file__).parent
    venv_dir = project_dir / "venv"

    print(f"Project directory: {project_dir}")
    print(f"Virtual environment will be created at: {venv_dir}")

    # Remove existing venv if it exists
    if venv_dir.exists():
        print("Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_dir)

    # Create virtual environment
    if not run_command([sys.executable, "-m", "venv", str(venv_dir)], "Creating virtual environment"):
        return 1

    # Determine activation script path
    if platform.system() == "Windows":
        activate_script = venv_dir / "Scripts" / "activate.bat"
        python_executable = venv_dir / "Scripts" / "python.exe"
        pip_executable = venv_dir / "Scripts" / "pip.exe"
    else:
        activate_script = venv_dir / "bin" / "activate"
        python_executable = venv_dir / "bin" / "python"
        pip_executable = venv_dir / "bin" / "pip"

    # Upgrade pip
    if not run_command([str(python_executable), "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip"):
        return 1

    # Install requirements
    requirements_file = project_dir / "requirements.txt"
    if requirements_file.exists():
        if not run_command([str(pip_executable), "install", "-r", str(requirements_file)], "Installing requirements"):
            return 1
    else:
        print("WARNING: requirements.txt not found!")
        return 1

    # Test installation
    print("\nTesting installation...")
    test_script = project_dir / "test_requirements.py"
    if test_script.exists():
        if not run_command([str(python_executable), str(test_script)], "Testing dependencies"):
            print("WARNING: Some dependencies failed to install properly")
            return 1

    print("\n" + "=" * 50)
    print("SUCCESS: Virtual environment setup completed successfully!")
    print("\nNext steps:")

    if platform.system() == "Windows":
        print(f"   1. Activate the virtual environment:")
        print(f"      {activate_script}")
        print(f"   2. Run the application:")
        print(f"      start.bat")
    else:
        print(f"   1. Activate the virtual environment:")
        print(f"      source {activate_script}")
        print(f"   2. Run the application:")
        print(f"      ./start.sh")

    return 0


if __name__ == "__main__":
    sys.exit(main())
