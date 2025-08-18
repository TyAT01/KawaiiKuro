#!/usr/bin/env python3
import sys
import os

# Ensure the 'kuro' directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from kuro.main import main

if __name__ == "__main__":
    main()
