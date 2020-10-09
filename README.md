# GPie
**G**aussian **P**rocess t**i**ny **e**xplorer (under construction)

- **simple**: an intuitive syntax that imitates scikit-learn
- **minimal**: a compact core of expressive abstractions
- **extensible**: a modular design for effortless composition


### Features

- several "avant-garde" kernels such as spectral kernel and neural kernel allow for exploration of new ideas
- each kernel implements anisotropic variant besides isotropic one to support automatic relevance determination
- a full-fledged toolkit of kernel operators enables all sorts of "kernel engineering", for example handcrafting composite kernels based on expert knowledge or exploiting special sturcture of datasets
- core computations such as likelihood and analytical gradient are carefully formulated for speed and robustness
- Bayesian optimizer offers a powerful strategy in optimizing expensive-to-evaluate, black-box objectives

### Functionalities

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
    - Kronecker sum (under construction)
    - Kronecker product (under construction)
- Gaussian process
    - regression
    - classification (under construction)
- t process (under construction)
    - regression
    - classification
- Bayesian optimizer
    - surrogate: Gaussian process, t process, etc
    - acquisition: expected improvement, lower confidence bound, etc

### Computational backend

- linear algebra: numpy
- optimization: scipy
- sampling inference: pymc3


### Examples
![alt text](./examples/mauna-loa-co2.png)