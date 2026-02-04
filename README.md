# OCL for QC

the code for generating 1 qubit drifting data is in `generate_stream.py`, by calling the function `generate_qubit_stream`. 

the model training code is in `cl.ipynb`. 

the visualization of the results are in `visualize.ipynb`, and the animation is in `qubit_drift.mp4`. 

the training accuracies for both strategies are in `naive.csv` and `replay.csv`. 

## Installing dependencies 

I used `uv`  to manage python packages. to install uv, use 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

to install the dependencies, first `git clone` the repository, then `cd` to the project folder and then run 

```bash
uv sync 
```

in terminal.  this will automatically creates a venv, install the correct python version `python 3.10` and install all the dependencies.
