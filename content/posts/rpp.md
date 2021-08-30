---
title: "Release planning problem" 
date: 2021-08-11T10:00:10-03:00
subtitle: ""
linksAuthors : {"/members/leonardo-hoet" : "Leonardo Hoet"}
image: ""
tags: ["optimization","cbc","rpp","linear programming"]
draft: false
---
A more powerful model than the next release problem which can provide a schedule of how a project need to plan its releases.


<!--more-->
# Release planning problem

As we have seen in a [previous blog post](https://giicis.github.io/posts/nrp/), the NRP problem can be very useful at the time of planning a release. But sometimes this is not enough, we may want to plan the whole project to have an idea of how it will evolve.
In this blog, we present the formal definition of the release planning problem which aims to solve the previous issue.

In a common language, the problem will be: Find the order in which a set of requirements should be implemented in order to maximize profit of a project given that we have different releases and stakeholders to satisfy.

To implement this model for a project, we'll need:
- Requirements in the project. For each one:
  - How much it will cost
- Number of stakeholders in the project. For each one of them we'll need:
  - The set of requirements in which the stakeholder has interest
  - A metric to measure the impact if the stakeholder is satisfied 
- Maximum cost the next release can have
- Number of planned releases



## Formal definition

- Let \\(k \in \mathbb{N}\\)  be releases to take into account.
- Let \\(R\\) be a set of requirements to be developed with \\(|R| = n\\)
- Let \\(S\\) be a set of stakeholders with \\(|S| = m\\)
- Let \\(X \in \lbrace 1,\dotsc,k+1\rbrace^n\\) be an array of integers that represents for each requirement \\(i\\) the number of the release in which it is implemented. If \\(x_i = k+1\\) the requirement \\(i\\) is not implemented
- Let \\(Y \in \lbrace 0,1\rbrace ^{(k+1) \cdot n}\\) be a matrix of binary variables, where \\(y_{l,i} \in Y / y_{l,i} = 1\\) if the requirement \\(i\\) is implemented in release \\(l\\)



$$
Y = \begin{bmatrix} y_{1,1} & \cdots & y_{1,n} \\\\\\ \vdots & \ddots & \vdots \\\\\\ y_{k+1,1} & \cdots & y_{k+1,n} \\\\\\ \end{bmatrix}
$$



- Let \\(E = [e_1,e_2,\dotsc,e_n]\\) be an array of efforts associated with each requirement
- Let \\(p\\) be the max affordable effort in each release
- Let \\(B= [b_1,b_2,\dotsc,b_m]\\) be an array where \\(b_i\\) indicates the profit if stakeholder \\(i\\) is satisfied.
- Let \\(P\\) be the precedence relation between \\((i,j)\\) where \\(i,j\\) are requirements; meaning that \\(i\\) requirement must be implemented if \\(j\\) requirement is implemented.
- Let \\(A \in \mathbb{R}^{m\cdot n}\\) be the priority matrix, where \\(a_{s,i} \in A / a_{s,i}\\) is the priority of the stakeholder \\(s\\) for a requirement \\(i\\). This matrix must be normalized, i.e. the sum of the elements for each row must be 1

The objective function (OF) is:

$$
max f(x) = \sum_{s = 1}^{m} \sum_{i=1}^{n} b_s \cdot  a_{s,i} \cdot (k+1-x_i)
$$

This function seems a little weird, so let's try to explain it. \\(a\\) and \\(b\\) are only parameters of the model. The interesting part is \\(k+1 - x_i\\) where if the requirement is implemented first, this part will be bigger and it will maximize the function taken into account its contribution to the whole model with \\(a\\) adn \\(b\\) as information. If \\(x_i = k+1\\) this part will be 0 and it won't affect de OF. 


subject to:
1) Release constraint: \\(x_i\\) must contain the number of release in which the requirement is implemented

$$
x_i = \sum_{l=1}^{k+1} l \cdot y_{li} \quad \forall i \in \{1,\dotsc, n\}
$$


2) Implementation constraint: every requirement should be implemented 

$$
\sum_{l=1}^{k+1} y_{li} = 1 \quad \forall i \in \{1,\dotsc ,n\}
$$

3) Effort constraint: The sum of the effort for each requirement in the release must be less or equal than the max affordable cost

$$
\sum_{i=1}^{n} e_i \cdot y_{li} \leq p \quad \forall l \in \{1, \dotsc ,k+1 \}
$$

4) Precedence constraint: requirement \\(i\\) must be implemented before or in the same release than requirement \\(j\\)

$$
x_i \leq x_j \quad \forall (i,j) \in P
$$

# Implementation

For the sake of simplicity, we'll using the same problem stated in the [previous blog post](https://giicis.github.io/posts/nrp/#fictional-problem) with the addition that  now we have to add the number of releases. Since it is a small problem let's use 2 releases.

Because we have defined that what is left in the release \\(k+1\\) is not implemented, the model is going to read `number_of_releases` as \\(k+1\\). So, in the data file we need `number_of_releases = 3` to have 2 releases. It sounds ugly but this simplifies the code.  

We need to add this to the file `data.dat`.
```
param number_of_releases := 3;
```


## Using pyomo

First, we import the needed libraries 
```python
# Import the needed libraries
from __future__ import division
import pyomo.environ as pyo
import math as mt
import sys
import numpy as np
```

Second, we need to define some functions that we'll help us to normalize the matrix A stated in [formal model](#formal-definition)

```python
# Helper functions used to normalize rpp.A
def normalize(x):
    """
    Given a numpy vector X, returns another vector the where the sum of elements is 1
    """
    acc = np.sum(x)
    if acc == 0:
        return x
    return x / acc


def A_normalizate(rpp):
    """
    Given an rpp model with A matrix
    Normalize each row, so the sum of elements per row is 1
    """
    A = np.zeros((rpp.number_of_stakeholders.value, rpp.number_of_requirements.value))
       
    # Assign rpp.A values to A
    for (i, j) in rpp.A.index_set():
        A[i - 1, j - 1] = rpp.A[i, j].value

    # Normalize A
    for i in range(0, A.shape[0]):
        A[i, :] = normalize(A[i, :])

    # Assign A values to rpp.A
    for j in range(0, A.shape[1]):
        for i in range(0, A.shape[0]):
            rpp.A[i + 1, j + 1] = A[i, j]
```

The code that is responsible of creating an abstract model:
```python
def abstract_model():
    """
    Creates an abstract model of Rpp problem
    """
    rpp = pyo.AbstractModel()

    # Model's parameters
    rpp.number_of_requirements = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.number_of_stakeholders = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.number_of_releases = pyo.Param(within=pyo.NonNegativeIntegers)
    rpp.max_cost = pyo.Param(within=pyo.NonNegativeIntegers, mutable=True)
    
    # Sets that will be used to iterate over 
    rpp.requirements = pyo.RangeSet(1, rpp.number_of_requirements)
    rpp.stakeholders = pyo.RangeSet(1, rpp.number_of_stakeholders)
    rpp.releases = pyo.RangeSet(1, rpp.number_of_releases)
    
    # Parameters defined over previous defined sets
    rpp.efforts = pyo.Param(rpp.requirements)
    rpp.profits = pyo.Param(rpp.stakeholders)
    
    # Relations defined over the cartesian product of sets
    # (i,j) requirement i should be implemented if j is implemented
    rpp.precedences = pyo.Set(within=rpp.requirements * rpp.requirements)
    # (s,i) > 0 if stakeholder s has interest over requirement i
    # This relation is here because the dataset have this information
    # We are using this to initialize matrix A
    rpp.interests = pyo.Set(within=rpp.stakeholders * rpp.requirements)

    # We use this function to assign a requirement priority for each stakeholder
    # This is because the dataset we are using does not have this information
    def A_init(rpp, s, i):
        if (s, i) in rpp.interests:
            return 1
        return 0
    # This parameter needs to be mutable so later on we can normalize it
    rpp.A = pyo.Param(rpp.stakeholders, rpp.requirements, initialize=A_init, mutable=True)

    # Variables
    # Store the number in which the requirement is implemented
    rpp.x = pyo.Var(rpp.requirements, domain=pyo.Integers)
    # y[l,i] == 1 if requirement i is implemented in l release
    rpp.y = pyo.Var(rpp.releases, rpp.requirements, domain=pyo.Binary)

    # Objective function
    def obj_function_rule(rpp):
        inner_sum = lambda s: sum(rpp.A[s, i] * (rpp.number_of_releases - rpp.x[i]) for i in rpp.requirements)
        return sum(rpp.profits[s] * inner_sum(s) for s in rpp.stakeholders)
        #return sum(rpp.profits[s] * sum(rpp.A[s, i] * rpp.number_of_releases - rpp.x[i] for i in rpp.requirements) for s in rpp.stakeholders)
    rpp.OBJ = pyo.Objective(rule=obj_function_rule, sense=pyo.maximize)

    # Constraints
    def release_constraint_rule(rpp, i):
        return sum(rpp.y[l, i] * l for l in rpp.releases) == rpp.x[i]
    rpp.release_constraint = pyo.Constraint(rpp.requirements, rule=release_constraint_rule)

    def implementation_constraint_rule(rpp, i):
        return sum(rpp.y[l, i] for l in rpp.releases) == 1
    rpp.implementation_constraint = pyo.Constraint(rpp.requirements, rule=implementation_constraint_rule)

    def effort_constraint_rule(rpp, l):
        return sum(rpp.efforts[i] * rpp.y[l, i] for i in rpp.requirements) <= rpp.max_cost
    rpp.efforts_constraint = pyo.Constraint(pyo.RangeSet(1, rpp.number_of_releases - 1), rule=effort_constraint_rule)

    def precedence(rpp, i, j):
        return rpp.x[i] <= rpp.x[j]
    rpp.precedences_constraint = pyo.Constraint(rpp.precedences, rule=precedence_constraint_rule)
    
    return rpp
```

Create, fill and solve the model described [above](#implementation).
```python
# This is the actual code that solves the problem

# Define the name of the solver to use
solver_name = 'cbc'
data_file = "./datasets/rpp_data.dat"

# Create the abstract model
rpp = abstract_model()
# Fill the model with concrete values
rpp_concrete = rpp.create_instance(data=data_file)
rpp_concrete.max_cost = 40

# Because we dont now what priority  stakeholders are going to assign to each requierement
# the normalization must be done with a concrete instance
A_normalizate(rpp_concrete)

# Create a new solver instance
solver = pyo.SolverFactory(solver_name)
if solver.name != 'glpk':
    # Assign 4 threads to the solver
    solver.options['threads'] = 4
# Solve the model and display the solution
res = solver.solve(rpp_concrete)
res['Solver'][0]['Status']
```


## Exploring the solution

We can ask pyomo information about the solution with `pyo.display`.

To see which requirements should be implemented in each release
 

```python
pyo.display(rpp_concrete.x)
```
```
x : Size=5, Index=requirements
    Key : Lower : Value : Upper : Fixed : Stale : Domain
      1 :  None :   1.0 :  None : False : False : Integers
      2 :  None :   2.0 :  None : False : False : Integers
      3 :  None :   3.0 :  None : False : False : Integers
      4 :  None :   3.0 :  None : False : False : Integers
      5 :  None :   2.0 :  None : False : False : Integers
```

Requirement 1 is the only one that can be implemented in the first release because it has a cost of 40 and our max cost is 40. In the second release requirements 2 and 5 will be implemented. Finally, requirements 3 and 4 are left to implement in the future.

If we raise the `max_cost` and solve the model again, we can see how the schedule changes:
 

```python
rpp_concrete.max_cost = 50
solver.solve(rpp_concrete)
pyo.display(rpp_concrete.x)
```
```text
x : Size=5, Index=requirements
    Key : Lower : Value : Upper : Fixed : Stale : Domain
      1 :  None :   1.0 :  None : False : False : Integers
      2 :  None :   2.0 :  None : False : False : Integers
      3 :  None :   2.0 :  None : False : False : Integers
      4 :  None :   3.0 :  None : False : False : Integers
      5 :  None :   1.0 :  None : False : False : Integers
```

Here, requirement 4 is left to implement in a future release.


# Conclusion 
As we have seen, this model provides a more powerful approach than the [next release problem](https://giicis.github.io/posts/nrp/).
Nonetheless, it still has drawbacks. Right now, we are assuming that the effort to implement a requirement is constant and it will not change during its implementation. In the software engineering world we know this is not true. Most of the time, a requirement needs more effort to complete. 
In the next blog post, we'll see how this problem can be modeled and solved using  fuzzy logic.