# MoS2_CTM_Simulation_Framework

Code and data for the manuscript **"Modelling and Simulation of Tri-Layer Al2O3/HfO2/Al2O3 MoS2 Charge Trap Memtransistors"**
(S. Pal, T. Gangwal, T. Mishra, P. Jaiswal, M. Raj, N. Goel; Department of Electronics and Communication Engineering,
Netaji Subhas University of Technology, New Delhi, India).

The package contains the physics-based device model of the tri-layer Al2O3/HfO2/Al2O3 MoS2 charge-trap memtransistor and
every calculation that produces a number, table or figure of the manuscript and its Supporting Information.
**The full guide is [README.docx](README.docx)**: what each script does, where its result appears in the paper, and how the check works.

## Layout

| Folder | Contents |
|---|---|
| `model/` | the device model of one cell |
| `device_studies/` | calibration of the model and single-cell physics |
| `applications/` | crossbar networks, Newton-Raphson solver, on-chip learning |
| `plotting/`, `figures/` | figure scripts and the figures of the paper |
| `data/measured/` | calibration data read from the cited articles, each with its source |
| `data/mnist/` | the MNIST images used for training and testing |
| `data/trained_networks/` | the networks trained for the paper |
| `results/`, `reference_results/` | working results and the reference copy used in the paper |
| `devsim/` | DEVSIM field calculation (results included) |

## Install and run

Python 3.11 (tested with 3.11.9).

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux or macOS
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` reruns the studies and compares every number with `reference_results/`. Long calculations (refits) are
rerun only with `python run_all.py --long`. See README.docx, Sections 3 to 6.

No value in this package was measured by the authors; the calibration data come from the cited articles (README.docx, Section 8).

## MNIST

The handwritten-digit images come from the MNIST database: Y. LeCun, C. Cortes and C. J. C. Burges,
*The MNIST database of handwritten digits*, available on Kaggle at https://www.kaggle.com/datasets/vikramtiwari/mnist-numpy (stored here as `data/mnist/mnist.npz` in the NumPy format).
