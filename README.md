# DiscreteZoo

This repository is a fork of both the [TextAttack](https://github.com/QData/TextAttack) and the [TextAttack-Search-Benchmark](https://github.com/QData/TextAttack-Search-Benchmark) repositories. The code in this repository is for the paper Don't Search for a Search Method, to appear at EMNLP 2021.

## Setup

### How to run

To reproduce all the experiments in the paper, run the python file `grid_run_all.py`. All results will be written to the folder `grid_results`.

### Attacks and how to design a new attack 

The `attack_one` method in an `Attack` takes as input an `AttackedText`, and outputs either a `SuccessfulAttackResult` if it succeeds or a `FailedAttackResult` if it fails. 

TextAttack formulates an attack as consisting of four components: a **goal function** which determines if the attack has succeeded, **constraints** defining which perturbations are valid, a **transformation** that generates potential modifications given an input, and a **search method** which traverses through the search space of possible perturbations. The attack attempts to perturb an input text such that the model output fulfills the goal function (i.e., indicating whether the attack is successful) and the perturbation adheres to the set of constraints (e.g., grammar constraint, semantic similarity constraint). A search method is used to find a sequence of transformations that produce a successful adversarial example.

The attacks for the paper are implemented in the folder `textattack/search_methods`.

#### Goal Functions

A `GoalFunction` takes as input an `AttackedText` object, scores it, and determines whether the attack has succeeded, returning a `GoalFunctionResult`.

#### Constraints

A `Constraint` takes as input a current `AttackedText`, and a list of transformed `AttackedText`s. For each transformed option, it returns a boolean representing whether the constraint is met.

#### Transformations

A `Transformation` takes as input an `AttackedText` and returns a list of possible transformed `AttackedText`s. For example, a transformation might return all possible synonym replacements.

#### Search Methods

A `SearchMethod` takes as input an initial `GoalFunctionResult` and returns a final `GoalFunctionResult` The search is given access to the `get_transformations` function, which takes as input an `AttackedText` object and outputs a list of possible transformations filtered by meeting all of the attack’s constraints. A search consists of successive calls to `get_transformations` until the search succeeds (determined using `get_goal_results`) or is exhausted.


## Citing our work

If you use our attacks or baselines for your research, please cite [placeholder]()

```bibtex
@misc{berger2021,
    title={Placeholder},
    author={Placeholder},
    year={2021},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

Because this work is based on TextAttack, if you use TextAttack for your research, please cite the original authors of the framework. [TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP](https://arxiv.org/abs/2005.05909).

```bibtex
@misc{morris2020textattack,
    title={TextAttack: A Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP},
    author={John X. Morris and Eli Lifland and Jin Yong Yoo and Jake Grigsby and Di Jin and Yanjun Qi},
    year={2020},
    eprint={2005.05909},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


