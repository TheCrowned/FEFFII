fen
python examples/feffi_lid-driven-cavity.py --nu 1e-2 --final-time 1e10 --steps-n 10 --store-sol -vv --plot-path "plots/LDC_1e-2_stepsn_10"
python examples/feffi_lid-driven-cavity.py --nu 5e-3 --final-time 1e10 --steps-n 10 --store-sol -vv --plot-path "plots/LDC_5e-3_stepsn_10"
python examples/feffi_lid-driven-cavity.py --nu 2.5e-3 --final-time 1e10 --steps-n 10 --store-sol -vv --plot-path "plots/LDC_2.5e-3_stepsn_10"
python examples/feffi_lid-driven-cavity.py --nu 1e-3 --final-time 1e10 --steps-n 10 --store-sol -vv --plot-path "plots/LDC_1e-3_stepsn_10"
python examples/feffi_lid-driven-cavity.py --nu 5e-4 --final-time 1e10 --steps-n 10 --store-sol -vv --plot-path "plots/LDC_5e-4_stepsn_10"
python examples/feffi_lid-driven-cavity.py --nu 2.5e-4 --final-time 1e10 --steps-n 10 --store-sol -vv --plot-path "plots/LDC_2.5e-4_stepsn_10"
python examples/feffi_lid-driven-cavity.py --nu 1e-4 --final-time 1e10 --steps-n 10 --store-sol -vv --plot-path "plots/LDC_1e-4_stepsn_10"

python examples/feffi_buoyancy-driven-cavity.py --beta 1000 --final-time 1e10 --steps-n 100 --store-sol -vv --plot-path "plots/BDC_1000_stepsn_100"
python examples/feffi_buoyancy-driven-cavity.py --beta 10000 --final-time 1e10 --steps-n 100 --store-sol -vv --plot-path "plots/BDC_10000_stepsn_100"
python examples/feffi_buoyancy-driven-cavity.py --beta 10000 --final-time 1e10 --steps-n 100 --store-sol -vv --plot-path "plots/BDC_100000_stepsn_100"
python examples/feffi_buoyancy-driven-cavity.py --beta 10000 --final-time 1e10 --steps-n 100 --store-sol -vv --plot-path "plots/BDC_1000000_stepsn_100"

python examples/feffi_buoyancy-driven-cavity.py --config-file "feffi/config/rayleigh-benard-convection.yml" --beta 1000 --final-time 1e10 --steps-n 100 --store-sol -vv --plot-path "plots/RBC_1000_stepsn_100" 
