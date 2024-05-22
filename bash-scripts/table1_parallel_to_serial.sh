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
find ../ex/benchmarks-deterministic/PRoTECT-versions -type f -exec sed -i '
  s/ result = parallel_dt_DS/ #result = parallel_dt_DS/g;
  s/ #result = dt_DS/ result = dt_DS/g;
  s/ result = parallel_ct_DS/ #result = parallel_ct_DS/g;
  s/ #result = ct_DS/ result = ct_DS/g
' {} +
