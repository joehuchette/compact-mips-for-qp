# Overview

This repository contains the implementation, driver, and analysis script for the paper ``Compact mixed-integer programming relaxations in quadratic optimization'' by Ben Beach, Robert Hildebrand, and Joey Huchette.

## Installation

1. Download [Julia v1](https://julialang.org/downloads/) and follow the installation instructions.
2. Download [Gurobi](https://www.gurobi.com/) and get a license file. Install Gurobi, activate your license, and then set the environment variable ``GUROBI_HOME`` pointing to XXX. For more information, see the [Gurobi.jl installation instructions](https://github.com/jump-dev/Gurobi.jl#installation).
3. Get a [MOSEK](https://www.mosek.com/) and move it to the default install location, e.g. ``$HOME/mosek/mosek.lic``.
4. Secure a [BARON](https://minlp.com/baron) binary and license. You will need to set the ``BARON_EXEC`` environment variable as discussed in the [BARON.jl installation instructions](https://github.com/joehuchette/BARON.jl#setting-up-baron-and-baronjl).
5. Download this repo (i.e. ``compact-mips-for-qp``). Define the ``MIP_FOR_QP_DIR`` environment variable pointing to the root directory of this repository (i.e. ``MIP_FOR_QP_DIR=/path/to/compact-mips-for-qp/``).
6. Start a Julia session. Activate a fresh environment with ``Pkg.activate("$(ENV[MIP_FOR_QP_DIR])")``. This should install all dependencies required for the experiments. If you encounter build errors with one of the solvers (e.g. ``Gurobi.jl``), please check that the environment variables are properly set and then manually attempt to build them (e.g. ``import Pkg; Pkg.build("Gurobi"))``).

## How to run the code

### Running the experiments
In the shell, run ``julia --project=$MIP_FOR_QP_DIR experiments/driver.jl``. You will need the proper environment variable ``MIP_FOR_QP_DIR'' defined as discussed above.

In order to run BARON with CPLEX as the LP/MIP solver, you will first need to have a working CPLEX install. Then, you can pass the ``CplexLibName`` parameter to BARON by editing the appropriate line in the ``baron_direct_factory`` function in ``common.jl``.

### Running the analysis
Run ``julia --project=$MIP_FOR_QP_DIR experiments/analysis.jl``.