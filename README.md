# PRoTECT

<p align="center">
<img src="./figs/JetEngine.png" alt="Example 1 - 2D Jet engine system (continuous-time deterministic system)" width="400"/>
<img src="./figs/VDP2.png" alt="Example 2 - 2D Jet engine system (continuous-time deterministic system)" width="400"/>
</p>

PRoTECT is an open-source software tool, for the parallelized
construction of safety barrier certificates (BCs) for nonlinear polynomial systems. This tool aims to verify safety properties for four classes of dynamical systems: (i) discrete-time stochastic systems (dt-SS), (ii) discrete-time deterministic systems (dt-DS), (iii) continuous-time stochastic systems (ct-SS), and (iv) continuous-time deterministic systems (ct-DS). PRoTECT is implemented in Python as an application programming interface (API), offering users the flexibility to interact either through its user-friendly graphic user interface (GUI) or via function calls from other Python programs. PRoTECT leverages parallelism across different barrier degrees to efficiently search for
a feasible BC. Additionally, PRoTECT employs sum-of-squares (SOS) optimization programs to systematically search for polynomial-type BCs.

We have provided Youtube tutorial videos to help understand how to use PRoTECT [here]().

## Table of Contents
- [Installation](#installation)
- [Examples](#examples)
- [Related Paper](#related-paper)
- [Reporting Bugs](#reporting-bugs)
- [License](#license)
- [Youtube Videos about PRoTECT]()

## Installation

We assume the user has python3 and pip installed on their machine. To install necessary dependencies, run from the directory containing the repository: <br><br>`pip install -r requirements.txt`

If you choose to use Mosek you will also need a license that can be acquired [here](https://www.mosek.com/license/request/?i=acp) (free for academics).

To use PRoTECT via its GUI, simply navigate a terminal to the current folder and then run `python3 main.py`. You can import pre-configured examples into the GUI from the folder [GUI_config_files](./ex/GUI_config_files/) by clicking the button *Import Config*. You can also run the examples for the [deterministic](./ex/benchmarks-deterministic/PRoTECT-versions/) and [stochastic](./ex/benchmarks-stochastic/) systems from the respective folders with `python3 <example-name>.py` (You may also need to temporarily add PRoTECT to your PATH using `export PYTHONPATH=/<path-to-PRoTECT>/PRoTECT:$PYTHONPATH` before running the examples this way, or permanently add it to your PATH by appending `export PYTHONPATH=$PYTHONPATH:/<path-to-PRoTECT>/PRoTECT` to the end of the file `~/.profile`, or equivalent, and restarting your computer). 

We have provided some tutorial videos which cover the basics of installation and using PRoTECT which can be found [here]().

## Examples

We present some selected examples graphically to demonstrate some use cases of PRoTECT. All the examples can be found in the folder [ex](./ex/) where the deterministic case studies also include the code to run them on the tool FOSSIL for comparison (the models in [models.py](./ex/benchmarks-deterministic/FOSSIL-versions/models.py) should be copied into the equivalent FOSSIL file models.py).

In addition configuration files for all of the examples can be imported in the GUI for analysis if desired, these can be found in the folder [GUI_config_files](./ex/GUI_config_files/).

### Example 1 - 2D Jet Engine (ct-DS)
<p align="center">
<img src="./figs/JetEngine.png" alt="Example 1 - 2D Jet engine system (continuous-time deterministic system)" width="400"/>
</p>
  
A continuous-time deterministic system of a 2D jet engine is verified over an infinite-time horizon with the goal of never reaching the red avoid region, see [ex2_jet_engine_ct_DS.py](./ex/benchmarks-deterministic/PRoTECT-versions/ex2_jet_engine_ct_DS.py).

### Example 2 - 2D Van der Pol oscillator (dt-SS)
<p align="center">
<img src="./figs/VDP2.png" alt="Example 2 - 2D Jet engine system (continuous-time deterministic system)" width="400"/>
</p>
  
A continuous-time deterministic system of a 2D Van der Pol oscillator is verified over a finite-time horizon with the goal of not reaching the red avoid region with some confidence, see [ex2_van_der_pol_oscillator_dt_SS_uniform.py](./ex/benchmarks-stochastic/ex2_van_der_pol_oscillator_dt_SS_uniform.py).

### Example 3 - 2D Two Room System (dt-DS)
<p align="center">
<img src="./figs/TwoRoom.png" alt="Example 3 - 2D Two room system (discrete-time deterministic system)" width="400"/>
</p>
  
A discrete-time deterministic system of a two-room temperature system that is verified over an infinite time horizon with the goal of never the red avoid region, see [ex2_TwoRoomTemp_dt_DS.py](./ex/benchmarks-deterministic/PRoTECT-versions/ex2_TwoRoomTemp_dt_DS.py).

### Examples 4 & 5 - 2D Linear and Nonlinear Systems (ct-SS)
<p align="center">
<img src="./figs/Linear2.png" alt="Example 2 - 2D linear system (continuous-time stochastic system)" width="400"/>
<img src="./figs/Nonlinear2.png" alt="Example 2 - 2D nonlinear system (continuous-time stochastic system)" width="400"/>
</p>
  
Two 2D continuous-time deterministic systems with Linear (left) and Nonlinear (right) dynamics are verified over a finite-time horizon with the goal of not reaching the red avoid region with some confidence, see [ex2_A1linear_ct_SS.py](./ex/benchmarks-stochastic/ex2_A1linear_ct_SS.py) and [ex2_nonlinear_ct_SS.py](./ex/benchmarks-stochastic/ex2_nonlinear_ct_SS.py).

## Related Paper

The arXiv version of the related paper will be published to arXiv imminently and is uploaded to this repository [here](./arXiv_PRoTECT.pdf).

### Authors
- [Ben Wooding](https://woodingben.com)
- [Viacheslav Horbanov](https://www.linkedin.com/in/slavixg/)
- [Abolfazl Lavaei](https://lavaei-cps.de/)

### Citing PRoTECT
```
To appear imminently
```

## Reporting Bugs
If you encounter any issues or have feedback, please open an issue in the repository. We appreciate your input and will address it as soon as possible.

## License
This project is licensed under the [MIT License](./LICENSE) see the file for details.
