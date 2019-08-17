Plonk examples
==============

Scripts
-------

There are two short example scripts to demonstrate Plonk. `analyze_disc.py` shows an example analysis of a accretion disc simulation. `visualize_disc.py` shows an example visualisation of a accretion disc.

Tutorial
--------

There is also a Jupyter notebook tutorial, saved in this repository as the markdown file `tutorial.md`. To convert it to a Jupyter notebook that can then be run you need Jupytext which can be installed with Conda, i.e.

```
conda install jupytext --channel conda-forge
```

Then to build the notebook, do

```
make notebook
```

This should produce a file `tutorial.ipynb` that you can run as a Jupyter notebook.

*Note: this requires both Jupyter and Jupytext to be installed. They both can be installed via Conda. See https://github.com/mwouts/jupytext for details on Jupytext.*