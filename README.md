# GPie
**G**aussian **P**rocess t**i**ny **e**xplorer

- **simple**: an intuitive syntax inspired by scikit-learn
- **minimal**: a compact core of expressive abstractions
- **extensible**: a modular design for effortless composition
- **lightweight**: as few dependencies as possible

This is a ongoing research project, and much of it is currently **under construction** - please expect bugs and sharp edges.


### Features

- several "avant-garde" kernels such as spectral kernel and neural kernel allow for exploration of new ideas
- each kernel implements anisotropic variant besides isotropic one to support automatic relevance determination
- a full-fledged toolkit of kernel operators enables all sorts of "kernel engineering", for example handcrafting composite kernels based on expert knowledge or exploiting special sturcture of datasets
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
    - elementwise sum
    - elementwise product
    - elementwise exponentiation
    - <span style="color:gray">Kronecker sum</span>
    - <span style="color:gray">Kronecker product</span>
- Gaussian process
    - regression
    - <span style="color:gray">classification</span>
- <span style="color:gray">t process</span>
    - <span style="color:gray">regression</span>
    - <span style="color:gray">classification</span>
- Bayesian optimizer
    - surrogate: Gaussian process, t process
    - acquisition: lower confidence bound, etc
- <span style="color:gray">sampling inference (via MCMC)</span>
    - <span style="color:gray">Metropolis-Hastings sampling</span>
    - <span style="color:gray">Hamiltonian Monte Carlo + no-U-turn auto-tuning</span>
- <span style="color:gray">variational inference</span>

Parts of the project <span style="color:gray">marked in grey color</span> are under construction.


### Examples

##### Gaussian process regression on Mauna Loa CO<sub>2</sub>
In this example, we use Gaussian process to model the concentration of CO<sub>2</sub> at Mauna Loa as a function of time.
![alt text](./examples/mauna-loa-co2.png)
In the plot, scattered dots represent historical observations, and shaded area shows the prediction interval made by a Gaussian process regressor trained on historical data.


### Installation
```bash
pip install --upgrade gpie
```
coming soon on pip...

### Backend

- numpy: linear algebra, stochastic sampling
- scipy: optimization, stochastic sampling
