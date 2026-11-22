# PriNCe analysis tool

This is a collection of tools to distribute calculations for UHECR propagation on a cluster and to collect and analyze the output. Developed mainly for use with the [PriNCe](https://github.com/joheinze/PriNCe) code.

These tools were developed for and used mainly in [Heinze et al., Astrophys.J. 873 (2019)](https://doi.org/10.3847/1538-4357/ab05ce)

## Software requirements

The majority of the code consists of pure Python modules.

Dependencies (list might be incomplete):

- Prince Ultra (`prince_ultra`, successor of the [*PriNCe* propagation code](https://github.com/joheinze/PriNCe));
  this fork no longer imports `prince` / `prince_cr`
- python-3.7 or later
- numpy
- scipy
- matplotlib
- iminuit
- jupyter notebook or jupyter lab (optional, but needed for the plotting example)
- Cluster running on Univa grid engine (for other clusters adjust `analyzer.cluster.template_submit` and all calls to `qsub` in `analyzer.cluster`)

## Basic usage

Adjust the paths and configs in `example_create_project.py`, then run as:

```bash
python example_create_project.py -c # create job folder and files
python example_create_project.py -s # submit all jobs
```

to check finished jobs:

```bash
python example_create_project.py -m #check missing job files
python example_create_project.py -m -s # resubmit missing jobs
```

to collect the project after all jobs are finished:

```bash
python example_create_project.py --collect
```

See `cluster.PropagationProject.run_terminal()`

To recompute only the fitting (and not the numerical propagation) see `python example_recompute_fit.py`. Call this file as:

```bash
python example_recompute_fit.py --fit -[options]
```

## Plotting fit results

The fit resutls are collected in `collected.hdf5`. This files contains the results in multi-dimensional numpy arrays, with dimensions corresponding to the shape of `config['paramlist']`. Utility functions for evalution are contained in `analyzer-plotter.py`. See `example_evaluate.ipynb` for example plots.

## Citation

If you are using this code in your work, please cite:

*A new view on Auger data and cosmogenic neutrinos in light of different nuclear disintegration and air-shower models*  
J. Heinze, A. Fedynitch, D. Boncioli and W. Winter  
[Astrophys.J. 873 (2019) no.1, 88](https://doi.org/10.3847/1538-4357/ab05ce)

## Author

Jonas Heinze

## Copyright and license

Copyright (c) 2020, Jonas Heinze All rights reserved.

licensed under BSD 3-Clause License (see LICENSE.md)
