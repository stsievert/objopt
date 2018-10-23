
This library provides...

1. theoretic values such as the strong convexity parameter
3. object-oriented definitions, both for models and optimization algorithms. This allows...
    * interacting with the optimization as an object. Want to compute some
      value partway through? Want to change the values as time goes on?
    * getting results intermediately (or in the presence of a keyboard
      interrept)
    * having callbacks, etc

## Features

This list is ongoing, I still need to implement some of these

### Problem formulations
https://en.wikipedia.org/wiki/Loss_functions_for_classification

* Least squares
* Logistic regression
* Hinge loss
* Cross entropy
* Squared error

### Regularization functions
* L2, L1 functions
* Nuclear norm?

### Algorithms
* Gradient descent
* SGD
    * naive SGD
    * with averaging (make citation)
    * changing batch size (make citation)
* SAGA
* FISTA, FASTA

A typical example:

``` python
def get_stats():
    # ...

model = Model()
opt = SGD(model.loss)


data = []
for _ in range(10):
    opt.step(steps=10)
    data += [get_stats(model)]
```

