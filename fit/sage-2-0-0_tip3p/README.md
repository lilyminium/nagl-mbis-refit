# Condensed-phase fitting

This directory contains the files needed to fit the forcefield inside `forcefield/force-field.offxml` to the physical property targets in `targets/phys-prop`. The targets here are the same as they were in the Sage 2.0 fit.

The `setup-up-forcefield.py` script writes the original `forcefield/force-field.offxml` file.
The `setup-options.py` file writes a file of some options to use for running Evaluator.
The `execute-fit-slurm-distributed.py` executes the ForceBalance fit with a SLURM backend, with additional utilities for renaming or removing persistent or temporary files, respectively.

`run-iris.sh` contains an example of running each of these scripts on a SLURM machine.

