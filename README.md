# GPie
[![Language](https://img.shields.io/github/languages/top/zackxzhang/gpie)](https://github.com/zackxzhang/gpie)
[![Python](https://img.shields.io/pypi/pyversions/gpie)](https://www.python.org)
[![PyPI](https://img.shields.io/pypi/v/gpie)](https://pypi.python.org/pypi/gpie)
[![License](https://img.shields.io/github/license/zackxzhang/gpie)](https://opensource.org/licenses/BSD-3-Clause)

**G**aussian **P**rocess t**i**ny **e**xplorer

- **simple**: an intuitive syntax inspired by scikit-learn
- **minimal**: a compact core of expressive abstractions
- **extensible**: a modular design for effortless composition
- **lightweight**: as few dependencies as possible

This is a ongoing research project with many parts currently **under construction** - please expect bugs and sharp edges.


### Features

- several "avant-garde" kernels such as spectral kernel and neural kernel allow for exploration of new ideas
- each kernel implements anisotropic variant besides isotropic one to support automatic relevance determination
- a full-fledged toolkit of kernel operators enables all sorts of "kernel engineering", for example handcrafting composite kernels based on expert knowledge or exploiting special structure of datasets
- core computations such as likelihood and analytical gradient are carefully formulated for speed and robustness
- Bayesian optimizer offers a powerful strategy in optimizing expensive-to-evaluate, black-box objectives


### Functionality

- kernel functions
    - white kernel
    - constant kernel
    - radial basis function kernel
    - rational quadratic kernel
    - Matérn kernel
        - Ornstein-Uhlenbeck kernel
    - periodic kernel
    - spectral kernel
    - neural kernel
- kernel operators
    - Hadamard (element-wise)
        - sum
        - product
        - exponentiation
    - *Kronecker*
        - *sum*
        - *product*
- Gaussian process
    - regression
    - *classification*
- *t process*
    - *regression*
    - *classification*
- Bayesian optimizer
    - surrogate: Gaussian process, *t process*
    - acquisition: lower confidence bound, etc
- sampling inference
    - Markov chain Monte Carlo
        - Metropolis-Hastings
        - *Hamiltonian*
        - *no-U-turn*
    - *simulated annealing*
- *variational inference*

Parts of the project *in italic font* are under construction.


### Examples

##### Gaussian process regression on Mauna Loa CO<sub>2</sub>
In this example, we use Gaussian process to model the concentration of CO<sub>2</sub> at Mauna Loa as a function of time.
![alt text](./examples/mauna-loa-co2.png)
In the plot, scattered dots represent historical observations, and shaded area shows the prediction interval made by a Gaussian process regressor trained on historical data.


### Installation

The fastest way to install GPie is from a prebuilt wheel using pip:
```bash
pip install --upgrade gpie
```

You can also install from source to try out the latest features (`pep517>=0.8.0` and `setuptools>=40.9.0` are needed):
```bash
pip install --upgrade git+https://github.com/zackxzhang/gpie
```


### Backend

- numpy: linear algebra, stochastic sampling
- scipy: optimization, stochastic sampling
