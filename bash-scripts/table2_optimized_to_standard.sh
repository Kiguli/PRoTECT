#!/bin/bash

# Function to check if a package is installed
is_installed() {
  dpkg -s "$1" &> /dev/null
  return $?
}

# Check if both packages are installed
need_update=false
if ! is_installed "findutils"; then
  need_update=true
  missing_packages="findutils"
fi

if ! is_installed "sed"; then
  need_update=true
  if [ -n "$missing_packages" ]; then
    missing_packages="$missing_packages sed"
  else
    missing_packages="sed"
  fi
fi

# Update package list and install missing packages if needed
if $need_update; then
  apt-get update
  apt-get install -y $missing_packages
fi

# Execute the find and replace command
find ../ex/benchmarks-stochastic -type f -exec sed -i "s/'optimize': True/'optimize': False/g" {} +
find ../ex/benchmarks-stochastic -type f -exec sed -i "s/'lam': 10,/'lam': None,/g" {} +
find ../ex/benchmarks-stochastic -type f -name "ex2_van_der_pol_oscillator_dt_SS_uniform.py" -exec sed -i "s/'lam': 1000,/'lam': None,/g" {} +
