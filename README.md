[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Build Status - GitHub](https://github.com/YairMZ/NR_telemetry_ECC/actions/workflows/python-app.yml/badge.svg)](
https://github.com/YairMZ/NR_Error_Correction/actions/workflows/python-app.yml/badge.svg)
[![codecov](https://codecov.io/gh/YairMZ/NR_telemetry_ECC/branch/master/graph/badge.svg?token=tEXXuFzTVz)](https://codecov.io/gh/YairMZ/NR_telemetry_ECC)


# NR Error Correction
Natural redundancy based error correction

My algorithms for error correction based / aided by natural redundancy in data.
The package considers natural redundancy which exists in sensor data such as telemetry data.
The protocol used to transmit the data also gives rise to additional redundancy.
The code currently considers only the MAVLink protocol, commonly used for UAV's.

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
 - Decoders - Implements structure based decoders. Subclass Decoder interface to create decoders.
 - Inference - Implements classes for general purpose inference, such as segmentation, or classification of buffers
 - [Utils](utils/README.md) - Includes various utilities.
 - Protocol Meta - Includes metadata about the protocol used.
 - MAVLink Utils - utility for generating new MAVLink dialects, as well as dialect used here. Based on
[pymavlink](https://github.com/ArduPilot/pymavlink).

-----------
## Other Directories
 - scripts - Includes scripts which should be used as "main" scrips to run full simulations.
 - examples - Used to hold small examples to showcase the use of some classes.


--------------------------
For questions or suggestions [contact me](mailto:yairmazal@gmail.com?subject=[GitHub]%20NR%20Error%20Correction).