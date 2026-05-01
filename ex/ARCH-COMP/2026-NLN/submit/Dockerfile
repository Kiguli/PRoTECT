FROM ubuntu:22.04
LABEL maintainer="Ben Wooding <ben.wooding@vanderbilt.edu>"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PRoTECT source pinned by build arg.
ARG PROTECT_COMMIT=main
RUN git clone https://github.com/Kiguli/PRoTECT.git /PRoTECT \
    && cd /PRoTECT && git checkout ${PROTECT_COMMIT}

WORKDIR /PRoTECT

# Solver dependencies (Mosek + CVXOPT both installed; PROTECT_SOLVER picks).
RUN pip3 install --no-cache-dir \
    Brian2==2.6.0 \
    colorama==0.4.6 \
    cvxopt==1.3.2 \
    Cython==3.0.10 \
    Jinja2==3.1.6 \
    MarkupSafe==2.1.5 \
    Mosek==10.2.0 \
    mpmath==1.3.0 \
    numpy==1.26.4 \
    packaging==24.0 \
    Pebble==5.0.7 \
    PICOS==2.4.17 \
    py-cpuinfo==9.0.0 \
    pyparsing==3.1.2 \
    SumOfSquares==1.2.2 \
    sympy==1.12 \
    scipy==1.11.4 \
    matplotlib==3.8.2

# MOSEK license. The .lic file lives in this submit/ folder locally and is
# blocked from git via .gitignore. The portal's build context contains it,
# so it gets baked into the image only on machines that already have the
# license -- never published to GitHub / GitLab.
COPY mosek.lic /licenses/mosek.lic
ENV MOSEKLM_LICENSE_FILE=/licenses/mosek.lic
ENV PROTECT_SOLVER=mosek

ENV PYTHONPATH=/PRoTECT

# Layer the packaged 2026-NLN benchmark suite (and new src/functions/
# helpers) onto the cloned PRoTECT tree.
COPY data /PRoTECT
# Output folder name MUST be 'results' (plural) -- the portal verifier
# checks for submit/results/results.csv exactly.
RUN mkdir -p /PRoTECT/ex/ARCH-COMP/2026-NLN/results

CMD ["python3", "/PRoTECT/ex/ARCH-COMP/2026-NLN/run_benchmarks.py"]
