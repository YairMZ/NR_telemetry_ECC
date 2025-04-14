[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status - GitHub](https://github.com/YairMZ/NR_telemetry_ECC/actions/workflows/python-app.yml/badge.svg)](
https://github.com/YairMZ/NR_Error_Correction/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/YairMZ/NR_telemetry_ECC/branch/main/graph/badge.svg?token=tEXXuFzTVz)](https://codecov.io/gh/YairMZ/NR_telemetry_ECC)


# NR Error Correction

Natural redundancy based error correction. Includes my algorithms for error correction based / aided by natural redundancy in data.
The package considers natural redundancy which exists in sensor data such as telemetry data.
The protocol used to transmit the data also gives rise to additional redundancy.
The code currently considers only the MAVLink protocol, commonly used for UAV's.

-----------

## Setup an Environment
- This code was only tested on Python 3.10.
- To run the code you need to install the required libraries. 
Preferably, set up a virtual environment using virtualenv or any other appropriate tool.
- After setting a virtual environment (or if you decide not to),  cd into the root directory and run:
```aiignore
pip install -r requirements.txt
```

## Run Tests
To run tests simply clone, cd into the cloned repo, and run:
```
python -m pytest
```
or
```
python -m pytest --cov-report=html
```
to run also coverage tests.

-----------

## Included Modules
 - [Decoders](decoders/README.md) - Implements structure based decoders. Subclass Decoder interface to create decoders.
 - Inference - Implements classes for general purpose inference, such as segmentation, or classification of buffers
 - [Utils](utils/README.md) - Includes various utilities.
 - Protocol Meta - Includes metadata about the protocol used.
 - MAVLink Utils - utility for generating new MAVLink dialects, as well as dialect used here. Based on
[pymavlink](https://github.com/ArduPilot/pymavlink).

-----------
## Other Directories
 - scripts - contains various special purpose scripts for massaging data and analysis.
 - runs - Scripts for running analysis either via simulation on actual experimental data.

-----------
## Running Simulations
- All the scripts in the runs directory are designed to be run from the command line.
- Run parameters are passed as command line arguments, and they affect the behavior of the simulation by altering:
  - The decoder type used in the simulation.
  - The decoder parameters.
  - The channel model used in the simulation.
  - The length of the simulation and more.
- I have simulation results (saved to Google Drive), saved with the parameters used to generate them.
- Consider beforehand that these simulations can take a long time to run, depending on:
  - the parameters you choose
  - the number of cores you have available
  - the amount of RAM you have available.
- Consider also that the simulation consumes quite a lot of RAM. If you run into memory issues, **it will crash**. A longer simualtion requires more RAM.

Within the runs directory, you can find three directories:
- [Feb_2022](runs/Feb_2022) - Contains scripts for comparing performance of decoders based on the experimental data collected in Feb 2022.
- [Feb_2023](runs/Feb_2023) - Contains scripts for comparing performance of decoders based on the experimental data collected in Feb 2023.
- [HC_eilat_July_2018](runs/HC_eilat_July_2018) - Contains scripts for comparing performance of decoders within simulation, i.e., channel errors are simulated. The data input into the simulation is based on the experimental data collected in Eilat, July 2018.



--------------------------
For questions or suggestions [contact me](mailto:yairmazal@gmail.com?subject=[GitHub]%20NR%20Error%20Correction).