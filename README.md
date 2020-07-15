## Envinronment setup
The easiest and quickest way of setting up an envinronment to run the code is through `conda`.
Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) and use the file `fenics_env.yml` to create a virtual environment already containing all the needed packages:
`conda env create -f fenics_env.yml`

When you want to run the code, make sure the environment is activated through `conda activate fenics_env`.

## Usage
usage: run-navier-stokes.py [-h] [--final-time FINAL_TIME] [--steps-n STEPS_N] [--precision SIMULATION_PRECISION] [--viscosity NU] [--rho-0 RHO_0] [--domain DOMAIN]
                            [--domain-size-x DOMAIN_SIZE_X] [--domain-size-y DOMAIN_SIZE_Y] [--shelf-size-x SHELF_SIZE_X] [--shelf-size-y SHELF_SIZE_Y]
                            [--mesh-resolution MESH_RESOLUTION] [--mesh-resolution-x MESH_RESOLUTION_X] [--mesh-resolution-y MESH_RESOLUTION_Y]
                            [--mesh-resolution-sea-top-y MESH_RESOLUTION_SEA_Y] [--store-sol] [--label LABEL] [-v] [-vv] [--plot | --no-plot]

optional arguments:
  -h, --help            show this help message and exit
  --final-time FINAL_TIME
                        How long to run the simulation for (hours) (default: 10)
  --steps-n STEPS_N     How many steps each of the "seconds" is made of (default: 1000)
  --precision SIMULATION_PRECISION
                        Precision at which converge is achieved, for all variables (power of ten) (default: -3)
  --viscosity NU        Viscosity, km^2/h (default: 0.36)
  --rho-0 RHO_0         Density, Pa*h^2/km^2 (default: 13230)
  --domain DOMAIN       What domain to use, either `square` or `custom` (default: custom)
  --domain-size-x DOMAIN_SIZE_X
                        Size of domain in x direction (i.e. width) (default: 1)
  --domain-size-y DOMAIN_SIZE_Y
                        Size of domain in y direction (i.e. height) (default: 1)
  --shelf-size-x SHELF_SIZE_X
                        Size of ice shelf in x direction (i.e. width) (default: 0.5)
  --shelf-size-y SHELF_SIZE_Y
                        Size of ice shelf in y direction (i.e. height) (default: 0.1)
  --mesh-resolution MESH_RESOLUTION
                        Mesh resolution (default: 10) - does not apply to `custom` domain
  --mesh-resolution-x MESH_RESOLUTION_X
                        Mesh resolution in x direction (default: 20) - only applies to `rectangle` domain
  --mesh-resolution-y MESH_RESOLUTION_Y
                        Mesh resolution in y direction (default: 4) - only applies to `rectangle` domain
  --mesh-resolution-sea-top-y MESH_RESOLUTION_SEA_Y
                        Mesh resolution for sea top beside ice shelf in y direction (default: 1) - only applies to `custom` domain
  --store-sol           Whether to save iteration solutions for display in Paraview (default: False)
  --label LABEL         Label to append to plots folder (default: )
  -v, --verbose         Whether to display debug info (default: True)
  -vv, --very-verbose   Whether to display debug info from FEniCS as well (default: False)
  --plot                Whether to plot solution (default: True)
  --no-plot             Whether to plot solution (default: True)

## Examples
- `python run-navier-stokes.py --domain square`
- `python run-navier-stokes.py --final-time 2 --steps-n 10000 --domain custom --domain-size-x 10 --mesh-resolution-x 100 --mesh-resolution-y 50 --mesh-resolution-sea-top-y 5 --precision -2`: will simulate 2 hours, each divided in 10000 time steps, in a domain 10x1 meters, with mesh resolution 100x50, with a y-resolution of 5 on sea top beside ice shelf, stopping when variables converge with precision 0.01