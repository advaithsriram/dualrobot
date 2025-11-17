"""
Main entry point for dual robot system.

Currently runs Robot A (UR5) with pick-and-place and trajectory execution.
This script can be extended to coordinate multiple robots.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np

# Import robot A module
import robotA

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main entry point for the simulation.
    Initializes and runs Robot A.
    """
    
    print("\n" + "="*70)
    print("DUAL ROBOT SYSTEM - MAIN CONTROLLER")
    print("="*70)
    print("Initializing Robot A (UR5)...\n")
    
    # Run Robot A
    robotA.main()


if __name__ == "__main__":
    main()
