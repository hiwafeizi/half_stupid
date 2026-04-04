"""Wrapper to run Rotterdam Minecraft builder from repo root.

Usage:
    python build_rotterdam.py --all              # Build all 10 in Minecraft
    python build_rotterdam.py euromast            # Build one
    python build_rotterdam.py --all --xml-only   # Just generate XML
    python build_rotterdam.py --list             # List buildings
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rotterdam_minecraft"))

from builder.malmo_build import main
main()
