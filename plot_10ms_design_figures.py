#!/usr/bin/env python3
"""Render the design-specified figures from the 10 ms experiment."""

from pathlib import Path

import plot_design_figures as figures


figures.RESULTS = Path("results_10ms/run_001")
figures.OUTPUT = Path("results_10ms/analysis")


if __name__ == "__main__":
    figures.main()
