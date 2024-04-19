# PRoTECT

PRoTECT is an open-source software tool, for the parallelized
construction of safety barrier certificates (BCs) for nonlinear polynomial systems. This tool aims to verify safety properties for four classes of dynamical systems: (i) discrete-time stochastic systems, (ii) discrete-time deterministic systems, (iii) continuous-time stochastic systems, and (iv) continuous-time deterministic systems. PRoTECT is implemented in Python as an application programming interface (API), offering users the flexibility to interact either through its user-friendly graphic user interface (GUI) or via function calls from other Python programs. PRoTECT leverages parallelism across different barrier degrees to efficiently search for
a feasible BC. Additionally, PRoTECT employs sum-of-squares (SOS) optimization programs to systematically search for polynomial-type BCs.

## Getting started

To install necessary dependencies, run: <br><br>`pip install -r requirements.txt`

If you choose to use Mosek you will also need a license that can be acquired [here](https://www.mosek.com/license/request/?i=acp) (free for academics).

To use PRoTECT via its GUI, simply navigate a terminal to the current folder and then run `python3 main.py`. You can import pre-configured examples into the GUI from the folder [GUI_config_files](./ex/GUI_config_files/) by clicking the button *Import Config*. You can also run the examples for the [deterministic](./ex/benchmarks-deterministic/PRoTECT-versions/) and [stochastic](./ex/benchmarks-stochastic/) systems from the respective folders with `python3 <example-name>.py` (You may also need to temporarily add this current directory to your PATH using `export PYTHONPATH=/../../PRoTECT:$PYTHONPATH` before running the examples this way). 

We have provided some tutorial videos which cover the basics of installation and using PRoTECT which can be found [here]().

## Related Paper

The related paper will be published to arXiv imminently and uploaded to this repository.

### Authors
- [Ben Wooding](https://woodingben.com)
- Viacheslav Horbanov
- [Abolfazl Lavaei](https://lavaei-cps.de/)

### Citing PRoTECT
```

```

## Feedback and Support
If you encounter any issues or have feedback, please open an issue in the repository. We appreciate your input and will address it as soon as possible.