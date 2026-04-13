#!/usr/bin/env bash
set -euo pipefail

files=(
    "integrated_quantities_plotting.py"
    "oscillation_analysis_plotting.py"
    "timescale_plotting.py"
    "appendix_plotting.py"
    "smhm_vcirc_over_redshift_plotting.py"
    "ism_gas_fraction_plotting.py"
    "CGM_fractions_plotting.py"
)

for f in "${files[@]}"; do
    echo "Running $f"
    python3 "$f" || echo "Error running $f"
done
