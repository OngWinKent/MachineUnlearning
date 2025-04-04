# Machine Unlearning

#### (Released on March 27, 2025)

## Introduction
This repository contains implementations of several popular machine unlearning algorithms in PyTorch, as listed below:

| Algorithm       | Paper                                                                                                                                               | Original Github Repository                            |
|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| gradient_ascent | [Unrolling SGD: Understanding Factors Influencing Machine Unlearning](https://arxiv.org/abs/2109.13398)                                             | https://github.com/cleverhans-lab/unrolling-sgd       |
| bad_teacher     | [Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using an Incompetent Teacher](https://arxiv.org/abs/2205.08096)                    | https://github.com/vikram2000b/bad-teaching-unlearning |
| scrub           | [Towards Unbounded Machine Unlearning](https://arxiv.org/abs/2302.09880)                                                                            | https://github.com/meghdadk/SCRUB/tree/main           |
| amnesiac        | [Amnesiac Machine Learning](https://arxiv.org/abs/2010.10981)                                                                                       | https://github.com/lmgraves/AmnesiacML                |
| boundary        | [Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the Decision Boundary](https://ieeexplore.ieee.org/abstract/document/10203289) | https://github.com/TY-LEE-KR/Boundary-Unlearning-Code |
| ntk             | [Forgetting Outside the Box: Scrubbing Deep Networks of Information Accessible from Input-Output Observations](https://arxiv.org/abs/2003.02960)    | https://github.com/AdityaGolatkar/SelectiveForgetting |
| fisher          | [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://arxiv.org/abs/1911.04933)                                     | https://github.com/AdityaGolatkar/SelectiveForgetting |
| unsir           | [Fast Yet Effective Machine Unlearning](https://arxiv.org/abs/2111.08947)                                                                           | https://github.com/vikram2000b/Fast-Machine-Unlearning|
| ssd             | [Fast Machine Unlearning Without Retraining Through Selective Synaptic Dampening](https://arxiv.org/abs/2308.07707)                                 | https://github.com/if-loops/selective-synaptic-dampening|

Sincere appreciation to the authors of these popular machine unlearning algorithms for open-sourcing their code, greatly contributing to the success of this repository.
## Getting started

### Preparation

Before executing the project code, please prepare the Python environment according to the `requirement.txt` file. We set up the environment with `python 3.9.12` and `torch 2.0.0`. 

```python
pip install -r requirement.txt
```

### How to run

**1. Model Training**

```python
python train_main.py -dataset Cifar10 
```


**2. Unlearning**

```python
python unlearn_main.py -gpu -dataset Cifar10 -unlearn_class 0 -unlearn_method retrain -model_path 
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the author by sending an email to
`winkent.ong at um.edu.my`.

# License and Copyright

The project is open source under BSD-3 license (see the `LICENSE` file).

©2024 Universiti Malaya.
