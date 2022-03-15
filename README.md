## Envinronment setup
The easiest and quickest way of setting up an envinronment to run the code is through `conda`.
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and use the file `fenics_env.yml` to create a virtual environment already containing all the needed packages:
`conda env create -f fenics_env.yml`

When you want to run the code, make sure the environment is activated through `conda activate fenics_env`.

You should also make sure that [Gmsh](http://gmsh.info/) is installed.

## Usage
Two main options:
1) The `feffi_` files in the `examples` run full simulations with some specific sets of parameters. Commandline arguments can be fed in as well, run `python examples/feffi_square.py -h` (for example) for a list of arguments and documentation. Config files can also be used.
Hit CTRL-C at any time during the simulation to quit it and have the results (up to that point) plotted. The plots folder opens up automatically.
2) Run python, `import feffi` and set up a simulation interactively.

## Parameters and units of measure
Parameter defaults are stored in `.yml` files inside the `config` directory. Provide a config file to feffi either through the `feffi.parameters.define_parameters()` function, or as a commandline arg `--config-file FILE`. Here we list units of measures in which the model expects the physical parameters to be, which is mostly SI units. However, it is possible to automatically convert all quantities from m/s to km/h by turning on `convert_from_ms_to_kmh` - this will speed up convergence. For a full list of parameters, refer to the Usage section.

- `final_time`: hours; total simulation time; default = 10.
- `steps_n`: integer, specifies into how many steps each of the simulation hours should be broken into (ex. `--steps-n 6` means an hour is split into 10-minutes chunks); default = 10000.
- `simulation_precision`: integer, power of ten at which convergence is achieved; default = -3.
- `nu`: m^2/s; kinematic viscosity; expects either 1, 2 or 4 entries depending on whether a scalar, vector or tensor is wanted. For tensor, entries should given in the order _xx, xy, yx, yy_ (OR IS IT?). Default = 100.
- `alpha`: diffusivity coefficient for temperature/salinity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished.
- `beta`: coefficient of thermal expansion; default = 10^-4 1/°C.
- `gamma`: coefficient of saline contraction; default = 7.6*10^-4 1/PSU.
- `T_0`: reference value for temperature in Bousinessq approximation; default = 1 °C.
- `S_0`: reference value for salinity in Bousinessq approximation; default = 35 PSU.
- `rho_0`: kg/m^3; base density of Bousinessq approximation; default = 1028.
- `domain_size_x`: km; domain width; default = 1.
- `domain_size_y`: km; domain height; default = 1.
- `mesh_resolution`: km; (average) distance between mesh nodes for unstructured mesh; default = 0.1.

For what concerns simulated quantities:

- velocity: km/h
- pressure: Pascal
- temperature: °C
- salinity: PSU

## Storing solutions
There are multiple solutions storage options.
By default, at the end of a simulation, the following is stored in the plot path:
- `config.yml`, containing values for all the config variables.
- `solutions`, containing mesh and xml solutions for velocity, temperature, salinity, melt rate.
- png plots of solutions.

Further storage options are controlled by apt settings:
- `store_solutions`: store solutions for velocity, pressure, temperature, salinity in a XDMF file. This happens with frequency controlled by `checkpoint_interval`.
- `checkpoint_interval`: controls how often solutions are stored in XDMF file. Png plots are also produced at every checkpoint.
- `store_daily_avg`: store daily averages for velocity, temperature, salinity, melt rate. These are stored in the `daily-avg` directory, with sub-dirs named with the day number.

## Reloading solutions
Given the plot path of a previous run, it is possible to reload its setup for further processing. For this to be possible, a plot path should at minimum contain a `config.yml` file and a `solutions` directory with the mesh+solutions `xml` files inside. If daily averages were stored, those will be retrieved as well.

To reload a status, use:

`f, domain, mesh, f_spaces, daily_avg = feffi.parameters.reload_status(plot_path)`

Solutions to the 3 equations system are not stored since they can be easily recomputed using the stored solutions:
```
meltparametrization.solve_3eqs_system(f)
(m_B, T_B, S_B) = f['3eqs']['sol'].split()
```

While the reloading is mostly meant for postprocessing purposes, it is in general also possible to restart the simulation from where it stopped. However, if you used a custom geometry/mesh, you will need to recreate it yourself. The mesh can be imported, but the subdomains will not, so BCs are unlikely to be applied correctly.

## Tests and benchmarks
As of today, benchmarks include `Lid Driven Cavity`, `Buoyancy Driven Cavity`, `Rayleigh-Benard Convection` and `Ford experiment`. These have been run with different parameters (i.e. with different _difficulty_), and we check that their result does not change when the model is updated. Simulations with precision 1e-5 have been run and saved in `feffi/reference-solutions`. There are a bunch of unit tests implemented in `unit_tests.py` that can be run which will check whether a newly computed solution matches the reference one.

To run a quick suite of basic benchmarks, use `python test/unit_tests.py FEFFIBenchmarksTestsQuick` (which include LDC nu 1e-2, BDC 100). For a more thorough set, use `python test/unit_tests.py FEFFIBenchmarksTestsThorough` (includes all the others).

Adding a new test is simple:
1. run the simulation you'd like to include as test. Run with a high simulation precision (like `-5`) and use the `plot_path` arg to give the plots directory a meaningful name.
2. when the simulation is over, copy the plots folder in `feffi/reference-solutions`. The directory should include a `solutions` subdir and a `config.yml` file (plus a `simul_data.csv`?).
3. add a new test in `unit_tests.py`.
4. commit the new `reference-solutions` dir and `unit_tests.py` (include pngs and `simulation.log` by appending the `-f` flag to `git add`, but exclude `solutions.h5` and `solutions.xdmf`).

It would be good practice to include all test simulations in `utilities/benchmarks.sh` so that they can be re-run if needed.
