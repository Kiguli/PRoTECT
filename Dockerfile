FROM ubuntu:22.04

# Set the working directory
WORKDIR /app

# Create necessary directories and copy files
RUN mkdir PRoTECT
COPY . ./PRoTECT

RUN cp ./PRoTECT/fossil-main.zip . && rm ./PRoTECT/fossil-main.zip

# Install dependencies
RUN apt-get update \
    && apt-get install -y python3 python3-pip curl nano unzip findutils sed \
    && rm -rf /var/lib/apt/lists/*

# Replace 'mosek' with 'cvxopt' in all files in /app/PRoTECT/ex
RUN find /app/PRoTECT/ex -type f -exec sed -i 's/mosek/cvxopt/g' {} +

# Unzip and clean up
RUN unzip fossil-main.zip && rm fossil-main.zip

# Append models.py content
RUN cat /app/PRoTECT/ex/benchmarks-deterministic/FOSSIL-versions/models.py >> /app/fossil-main/experiments/benchmarks/models.py

# Install PRoTECT dependencies
RUN cd PRoTECT && pip install -r requirements.txt

# Install prerequisites for dreal4
RUN curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh' | bash

# Install fossil-main
RUN cd fossil-main && pip3 install .

# Set environment variables
ENV PYTHONPATH="/app/PRoTECT:/app/fossil-main"

# Create an entrypoint script to set the environment variables and start bash
RUN echo '#!/bin/bash\nexport PYTHONPATH="${PYTHONPATH}:/app/PRoTECT:/app/fossil-main"\nexec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
