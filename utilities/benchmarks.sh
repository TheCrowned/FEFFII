fen
python examples/feffi_lid-driven-cavity.py --nu 1e-2 --store-sol --plot-path "LDC_1e-2"
python examples/feffi_lid-driven-cavity.py --nu 5e-3 --store-sol --plot-path "LDC_5e-3"
python examples/feffi_lid-driven-cavity.py --nu 2.5e-3 --store-sol --plot-path "LDC_2.5e-3"
python examples/feffi_lid-driven-cavity.py --nu 1e-3 --store-sol --plot-path "LDC_1e-3"
python examples/feffi_lid-driven-cavity.py --nu 5e-4 --store-sol --plot-path "LDC_5e-4"
python examples/feffi_lid-driven-cavity.py --nu 2.5e-4 --store-sol --plot-path "LDC_2.5e-4"

python examples/feffi_buoyancy-driven-cavity.py --beta 1000 --store-sol --plot-path "BDC_1000"
python examples/feffi_buoyancy-driven-cavity.py --beta 10000 --store-sol --plot-path "BDC_10000"
python examples/feffi_buoyancy-driven-cavity.py --beta 100000 --store-sol --plot-path "BDC_100000"
python examples/feffi_buoyancy-driven-cavity.py --beta 1000000 --steps-n 500 --store-sol --plot-path "BDC_1000000"

python examples/feffi_buoyancy-driven-cavity.py --config-file "feffi/config/rayleigh-benard-convection-slip.yml" --beta 2500 --steps-n 350 --store-sol --plot-path "RBC_2500_slip"
python examples/feffi_buoyancy-driven-cavity.py --config-file "feffi/config/rayleigh-benard-convection-noslip.yml" --beta 2500 --steps-n 350 --store-sol --plot-path "RBC_2500_noslip"

# difficult ones, avoid hanging at the beginning
python examples/feffi_lid-driven-cavity.py --nu 1e-4 --store-sol --stab --plot-path "LDC_1e-4"
