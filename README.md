# CynLr Dual Robot Simulation

## How to Run

To run the main dual-robot simulation:

```bash
cd scripts
python main.py
```

Make sure you have all required dependencies installed (see below).

## Scripts Folder Structure

The `scripts` folder contains all the main code for the dual-robot simulation:

```
scripts/
├── main.py                # Main script for dual robot simulation
├── robotA.py              # UR5 robot setup, trajectory generation pick-and-place
├── robotB.py              # Franka Panda setup, camera functions
├── vision_processor.py    # Vision multiprocessing, image/detection queues
requirements.txt       # Python dependencies
```

## Robot Models (URDF)

URDF files for the robot models (UR5 and Franka Panda) are stored in the `urdf/` folder. These files are required for loading the robot models in the simulation.

## Requirements & Installation

All required Python packages are listed in `requirements.txt`.
To install them, run:

```bash
pip install -r requirements.txt
```

Typical requirements include:
- pybullet
- numpy
- matplotlib
- opencv-python

## Notes
- The simulation requires a GUI environment for PyBullet visualization and video recording.
- Output plots and videos are saved in the `graphs` directory.
