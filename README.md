# GPie
**G**aussian **P**rocess t**i**ny **e**xplorer (under construction)


Principles

- simple: an intuitive syntax that imitates scikit-learn
- minimal: a compact core of expressive abstractions
- extensible: a modular design for effortless composition


Functionalities

- kernel functions
    - white kernel
    - constant kernel
    - radial basis function kernel
    - rational quadratic kernel
    - Matérn kernel
    - Ornstein–Uhlenbeck kernel
    - periodic kernel
    - spectral kernel
    - neural kernel
- kernel operators
    - elementwise sum
    - elementwise product
    - elementwise exponentiation
    - Kronecker sum (under construction)
    - Kronecker product (under construction)
- Gaussian process regressor
- Gaussian process classifier (under construction)
- t process regressor (under construction)
- t process classifier (under construction)
- Bayesian optimizer


Features

- a couple of "avant-garde" kernels such as spectral kernel allow for exploration of new ideas
- each kernel implements anisotropic variant besides isotropic one to support automatic relevance determination
- a full-fledged toolkit of kernel operators enables all sorts of "kernel engineering", for example handcrafting composite kernels based on expert knowledge or exploiting special sturcture of datasets
- core computations like marginal likelihood and analytical gradients are carefully formulated for speed and robustness
- Bayesian optimizer offers a powerful strategy in optimizing expensive-to-evaluate, black-box objectives


Computational backend

- linear algebra: numpy
- optimization: scipy
- sampling inference: pymc3
