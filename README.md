## Envinronment setup
The easiest and quickest way of setting up an envinronment to run the code is through `conda`.
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and use the file `fenics_env.yml` to create a virtual environment already containing all the needed packages:
`conda env create -f fenics_env.yml`

When you want to run the code, make sure the environment is activated through `conda activate fenics_env`.

You should also make sure that [Gmsh](http://gmsh.info/) is installed.

## Usage
Two main options:
1) The `feffi_` files in the `examples` run full simulations with some specific sets of parameters. Commandline arguments can be fed in as well, run `python examples/feffi_square.py -h` (for example) for a list of arguments and documentation.
Hit CTRL-C at any time during the simulation to quit it and have the results (up to that point) plotted. The plots folder opens up automatically.
2) Run python, `import feffi` and set up a simulation interactively.

## Examples
- `python examples/feffi_square.py`
- `python examples/feffi_fjord.py --final-time 2 --steps-n 10000 --domain-size-x 10 --mesh-resolution-x 100 --mesh-resolution-y 50 --mesh-resolution-sea-top-y 5 --precision -2`: will simulate 2 hours, each divided in 10000 time steps, in a domain 10x1 meters, with mesh resolution 100x50, with a y-resolution of 5 on sea top beside ice shelf, stopping when variables converge with precision 0.01

## Parameters and units of measure
Parameter defaults are stored in `.yml` files inside the `config` directory. Provide a config file to feffi either through the `feffi.parameters.define_parameters()` function, or as a commandline arg `--config-file FILE`. Here we list units of measures in which the model expects the physical parameters to be, which is mostly SI units. However, some conversions happen in the code to make aid and speed up convergence. For a full list of parameters, refer to the Usage section.

- `final_time`: hours; total simulation time; default = 10.
- `steps_n`: integer, specifies into how many steps each of the simulation hours should be broken into (ex. `--steps-n 6` means an hour is split into 10-minutes chunks); default = 10000.
- `simulation_precision`: integer, power of ten at which convergence is achieved; default = -3.
- `g`: gravitational acceleration; default = 9.81.
- `nu`: m^2/s; kinematic viscosity; expects either 1, 2 or 4 entries depending on whether a scalar, vector or tensor is wanted. For tensor, entries should given in the order _xx, xy, yx, yy_ (OR IS IT?). Default = 100.
- `alpha`: diffusivity coefficient for temperature/salinity, m^2/s. Expects 1, 2 or 4 space-separated entries, depending on whether a scalar, vector or tensor is wished.
- `beta`: coefficient of thermal expansion; default = 10^-4 1/°C.
- `gamma`: coefficient of saline contraction; default = 7.6*10^-4 1/PSU.
- `T_0`: reference value for temperature in Bousinessq approximation; default = 1 °C.
- `S_0`: reference value for salinity in Bousinessq approximation; default = 35 PSU.
- `rho_0`: kg/m^3; base density of Bousinessq approximation; default = 1028.
- `domain_size_x`: km; domain width; default = 1.
- `domain_size_y`: km; domain height; default = 1.
- `shelf_size_x`: km; ice shelf width; default = 0.5.
- `shelf_size_y`: km; ice shelf heigth; default = 0.1.
- `mesh_resolution`: km; (average) distance between mesh nodes for unstructured mesh; default = 0.1.
- `mesh_resolution_x`: km; distance between mesh nodes x-wise; default = 0.1.
- `mesh_resolution_y`: distance between mesh nodes y-wise; default = 0.1.
- `mesh_resolution_sea_top_y`: distance between mesh nodes y-wise beneath the ice shelf; default = 0.1.

For what concerns simulated quantities:

- velocity: km/h
- pressure: Pascal
- temperature: °C
- salinity: PSU

## TODO
### Short term (comments by CH)
- Double check, for real, how the tensors of `assemble_viscosity_tensor` should be formulated. This was incorrect (I assume) for the scalar case, and have been ad-hoc extended to the vector and tensor case by CH. SO must finalize.
- Implement that the user can set `DirichletBC`s. Probably not as a part of the smoother command-line arguments, but if one would make your own `.py` file that contains a configuration. That is, by creating a mesh, getting function spaces, and then applying "custom" Dirichlet BCs.
- I don't see a reason why `boundaries/` and `configs/` should *not* be part of the module, rathern than being external as they are now.
- Is is good logic to have `--steps-n` to be "per unit time". If so (and I don't have a strong oppinion about this), make this very clear in the help, maybe even change the argument name. To me, if I see `--final-time` and `--steps-n`, I kinda put them together.

### Long term (no stress) 
- **Clean up code**. I have started this in `parameters.py` to show a bit, but it is not complete. To me, this makes the code more organized, easier to read and troubleshoot. I have, to some extent, applied the below points:
    + Start using a syntax checker that conforms to PEP standards. I think I use `autopep8`, but that is kinda a personal choice.
    + Make sure to have lines that don't exceed pep8 standard (<79 chars, or whatever)
    + Use `.format` rather than the old `%` sytax, i.e. `"Hello {:s} {:4.2f}".format('Christian', 0.15654)`
    + Check over some superflous logic and statments.
    + Maybe we could use the `fenics` class `Parameters` for the config (once it is inside `feffi`). Not sure this is the smoothest, so take it with a grain of salt. Will check out.
    + For saving, a very nice feature would be to:

        1. Save the config settings that were used (by writing the configs to file just after simulation).
        2. Adding to the saved configuration the git commit that this was done under and if it is modified.
		This way, one could (almost) always go back to a particular simulation.
