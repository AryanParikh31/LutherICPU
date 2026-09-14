"""
lutherICPU Standalone CLI Entry Point.

Usage:
  python luther_cli.py --images "F:/tandt_db/tandt/truck/images" --colmap "F:/tandt_db/tandt/truck/sparse/0" --iterations 30000 --serve
"""
import sys
from run_luther import main

if __name__ == "__main__":
    main()
