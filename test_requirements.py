#!/usr/bin/env python3
"""
Test script to verify all required dependencies are properly installed
and can be imported successfully.
"""

import sys
import importlib
from typing import List, Tuple


def test_import(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        if package_name:
            module = importlib.import_module(module_name, package_name)
        else:
            module = importlib.import_module(module_name)
        return True, f"[OK] {module_name} imported successfully"
    except ImportError as e:
        return False, f"[FAIL] Failed to import {module_name}: {e}"
    except Exception as e:
        return False, f"[ERROR] Error importing {module_name}: {e}"


def main():
    """Test all required dependencies."""
    print("Testing AnantaAI Dependencies")
    print("=" * 50)

    # Core dependencies
    dependencies = [
        # Core ML/AI
        ("torch", None),
        ("transformers", None),
        ("sentence_transformers", None),
        ("faiss", None),
        ("numpy", None),

        # Web framework
        ("fastapi", None),
        ("uvicorn", None),
        ("pydantic", None),

        # Web scraping
        ("requests", None),
        ("bs4", None),  # beautifulsoup4

        # Streamlit
        ("streamlit", None),

        # Utilities
        ("tqdm", None),
        ("typing_extensions", None),
    ]

    passed = 0
    failed = 0

    for module_name, package_name in dependencies:
        success, message = test_import(module_name, package_name)
        print(message)
        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("SUCCESS: All dependencies are properly installed!")
        return 0
    else:
        print("WARNING: Some dependencies are missing. Please install them using:")
        print("   pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
