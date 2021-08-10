---
title: "Next release problem"
date: 2021-08-08T23:02:38-03:00
subtitle: ""
linksAuthors : {"/members/leonardo-hoet" : "Leonardo Hoet"}
image: ""
tags: ["optimization", "cbc"]
draft: false
---


A common problem in every software project is which requirement should the team implement in order to satisfy stakeholders' needs. In this blog, we aim to define a formal model that can help us with this problem.

<!--more-->
# Introduction
As we have said, a big problem in every software development project is to determine a set of requirements which satisfies all parts involved (stakeholders). The next release problem (NRP) provides a formal mathematical model to this problem. This problems aims to find a subset of requirements that optimize a wanted attribute, such as profit or cost, given that we have to fulfill stakeholders' needs.
NRP can be reduced to the [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem) which, in the end, is a NP-Hard problem.

To implement this model we will need the following information
- Requirements in the project. For each one:
  - How much it will cost
- Number of stakeholders in the project. For each one of them we'll need:
  - The set of requirements in which the stakeholder has interest
  - A metric to measure the impact if the stakeholder is satisfied 
- Maximum cost the next release can have

## Formal definition of NRP Mono-Objective

The problem can be addressed as a mixed integer linear programming (MILP) model. The next model assumes that the decision maker wants to maximize the profit subject to a cost restriction. 


- Let \\(S\\) be the set of stakeholders with \\(|S| = n\\)
- Let \\(R\\) be the set of requirements with \\(|R| = m\\)
- Let  \\(p\\)  be the maximum affordable cost
- Let  \\(X = [x_1 , x_2 , … ,x_n ]\\)  be a binary array where the value of the cell \\( i = 1\\) if  requirement \\(i\\) is implemented. \\( 0 \\) otherwise 
- Let  \\(Y = [y_1, y_2,…,y_m]\\)  be a binary array where the value of the cell \\( j = 1\\) if the  stakeholder \\(j\\) is satisfied (that means, all of his requirements are implemented in the next release). \\( 0 \\) otherwise.
- Let  \\(C = [c_1, c_2, …, c_n ]\\)  be an array of cost per requirement.
- Let \\(B = [b_1, b_2 ,… ,b_m]\\) be the profit per stakeholder if it is satisfied.
- Let \\(P\\) be the precedence relation between \\((i,j)\\) where \\(i,j\\) are requirements; meaning that  requirement \\(i\\) must be selected if requirement \\(j \\)is selected. i.e. To implement \\(j\\), first implement \\(i\\)
- Let \\( I\\)  be the interest relation \\((i,k)\\) where  stakeholder \\(k\\) has interest over  requirement \\(i\\).


With this parameters, the model looks like:

$$
max \ f(Y) = \sum_{i \in S} b_i \cdot y_i
$$


subject to
1) A cost restriction. The cost of implementing each requirement needs to be less than the maximum affordable cost
$$
\sum_{j \in R} c_j \cdot x_j \leq p
$$



2) A precedence restriction. If requirement \\(x_j\\) needs to be implemented (i.e. \\(x_j = 1\\) \\(x_i\\) must be equal to 1 in order to not violate this restriction
$$
x_i \geq x_j \quad \forall (i,j) \in P
$$



3) An interest constraint. This restriction is used to set \\(y_k\\) if requirement \\(x_i\\) is implemented. Since the objective function is a maximization, if \\(x_i\\) is implemented, \\(y_k\\) is automatically  set to 1 because it maximize the objective.
$$
x_i \geq y_k \quad \forall (i,k) \in I
$$



4) Binary constraints
$$
X \in \lbrace0,1\rbrace^n
$$
$$
Y \in \lbrace0,1\rbrace^n
$$


# Implementing NRP in python
We'll be using python to find an optimal solution of this problem. Python has an excellent collection of libraries to model the problem. Moreover, these libraries are capable of calling  low level solvers such as CBC or CPLEX to solve the model. So python's speed will not affect us, since these solvers are written in C/C++. 

We are renaming the parameters in order to achieve a more readable code.

First, let's begin with importing the needing libraries

```python
import pyomo.environ as pyo
from pyomo.environ import AbstractModel
```

[Pyomo](http://www.pyomo.org/) is a library that provide common language for modelling and is able to call different solvers.

```python
# Create and abstract model
nrp_abs = pyo.AbstractModel()

# Assign parameters to the model
nrp_abs.number_of_requirements = pyo.Param(within=pyo.NonNegativeIntegers)
nrp_abs.number_of_costumers = pyo.Param(within=pyo.NonNegativeIntegers)
nrp_abs.max_cost = pyo.Param(within=pyo.NonNegativeIntegers, mutable=True)

# Sets used to maintain data of customers and requirements

nrp_abs.requirements = pyo.RangeSet(1, nrp_abs.number_of_requirements)
nrp_abs.customers = pyo.RangeSet(1, nrp_abs.number_of_costumers)


# Parameters for the model
nrp_abs.profit = pyo.Param(nrp_abs.customers)  # Profit of each customer if it is satisfied
nrp_abs.cost = pyo.Param(nrp_abs.requirements)  # Cost of implementing each requirement


# (i,j) requirement i should be implemented if j is implemented
# Set is within the cross product of Requirements X Requirements
nrp_abs.prerequisite = pyo.Set(within=nrp_abs.requirements * nrp_abs.requirements)
# (i,k) this relation exists if stakeholder k has interest on requirement i
nrp_abs.interest = pyo.Set(within=nrp_abs.customers * nrp_abs.requirements)

# Creation of variables
# x = 1 if requirement i is implemented in the next release, otherwise 0
nrp_abs.x = pyo.Var(nrp_abs.requirements, domain=pyo.Binary)
# y = 1 if all customer requirements are satisfied in the next release, otherwise 0
nrp_abs.y = pyo.Var(nrp_abs.customers, domain=pyo.Binary)

# Objective function
def obj_expression(nrp: AbstractModel):
    # Model should maximize profit of the next release
    return pyo.summation(nrp.profit, nrp.y)
nrp_abs.OBJ = pyo.Objective(rule=obj_expression, sense=pyo.maximize)


# Definition of cost constraint rule
def cost_constraint_rule(nrp: AbstractModel):
    # Cost should be keep under a predefined cost
    return pyo.summation(nrp.cost, nrp.x) <= nrp.max_cost
nrp_abs.cost_constraint = pyo.Constraint(rule=cost_constraint_rule)


# Defection of precedence constraint
def precedence_constraint_rule(nrp: AbstractModel, i: int, j: int):
    return nrp.x[i] >= nrp.x[j]
nrp_abs.precedence_constraint = pyo.Constraint(nrp_abs.prerequisite, rule=precedence_constraint_rule)

# Definition of interest constraint
# Each tuple in nrp.dat.interest is inverted, so the constraint is also inverted
def interest_constraint_rule(nrp: AbstractModel, i: int, k: int):
    return nrp.y[i] <= nrp.x[k]
nrp_abs.interest_constraint = pyo.Constraint(nrp_abs.interest, rule=interest_constraint_rule)
```

Now, we have an abstract model. This is very powerful, we can provide different datasets to see how the model performs.

Lets fill the model with actual data and solve it.

```python
data_file_path = '../datasets/nrp_100c_140r.dat'
nrp_concrete = nrp_abs.create_instance(data_file_path)
```


## Solving the model
We are using [CBC](https://github.com/coin-or/Cbc) because it's an open source and fast, but if you desire you can use different solvers.


```python
from pyomo.environ import SolverFactory
nrp_concrete.max_cost = 100
solver = SolverFactory('cbc')
res = solver.solve(nrp_concrete)
res.write()
```



```text
# ==========================================================
# = Solver Results                                         =
# ==========================================================
# ----------------------------------------------------------
#   Problem Information
# ----------------------------------------------------------
Problem: 
- Name: unknown
  Lower bound: 563.0
  Upper bound: 563.0
  Number of objectives: 1
  Number of constraints: 375
  Number of variables: 228
  Number of binary variables: 240
  Number of integer variables: 240
  Number of nonzeros: 100
  Sense: maximize
# ----------------------------------------------------------
#   Solver Information
# ----------------------------------------------------------
Solver: 
- Status: ok
  User time: -1.0
  System time: 0.15
  Wallclock time: 0.16
  Termination condition: optimal
  Termination message: Model was solved to optimality (subject to tolerances), and an optimal solution is available.
  Statistics: 
    Branch and bound: 
      Number of bounded subproblems: 8
      Number of created subproblems: 8
    Black box: 
      Number of iterations: 353
  Error rc: 0
  Time: 0.17249560356140137
# ----------------------------------------------------------
#   Solution Information
# ----------------------------------------------------------
Solution: 
- number of solutions: 0
  number of solutions displayed: 0
```
As you can see, CBC has found an optimal solution in 0.17 seconds.


## Exploring the solution
We can see the whole model using `pyo.display(nrp_concrete)` or we can use `pyo.display` on the parts  where we are interested


If we print the objective function, we'll get:
```python
# Print the value of the objective function
pyo.display(nrp_concrete.OBJ)
```


```text
OBJ : Size=1, Index=None, Active=True
    Key  : Active : Value
    None :   True : 985.0
```
That means, the optimal solution for this model costs 985 monetary units.


If we want to see which requirements should be implemented, we have to display the variables associated with them.

```python
# Display of variables associated to requirements
# Value == 1 means the requirement is implemented
pyo.display(nrp_concrete.x)
```

```text
Variables:
    x : Size=140, Index=requirements
        Key : Lower : Value : Upper : Fixed : Stale : Domain
          1 :     0 :   0.0 :     1 : False : False : Binary
          2 :     0 :   1.0 :     1 : False : False : Binary
          3 :     0 :   1.0 :     1 : False : False : Binary
          4 :     0 :   0.0 :     1 : False : False : Binary
          5 :     0 :   0.0 :     1 : False : False : Binary
          6 :     0 :   0.0 :     1 : False : False : Binary
          7 :     0 :   1.0 :     1 : False : False : Binary
          8 :     0 :   0.0 :     1 : False : False : Binary
```
We have cut down the output to make it more readable.
In this output, the variables with value of 1 are the ones that should be implemented. 

# Conclusion

This model can be very helpful in an environment with multiple stakeholders and multiple requirements. It can help
us reducing the complexity of making a decision. But what if we want to plan the releases for the whole project? In next blog posts we'll address this problem with another model.
