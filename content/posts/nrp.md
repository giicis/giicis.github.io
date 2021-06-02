---
title: "Next release problem"
date: 2021-05-28T23:02:38-03:00
subtitle: ""
linksAuthors : {"/members/leonardo-hoet" : "Leonardo Hoet"}
image: ""
tags: ["optimization", "cbc",]
draft: false
---

NRP problem are a type of problem...

<!--more-->
# Problem definition
A big problem in every software development project is to determine a set of requirements which satisfies all parts involved (stakeholders). The next release problem (NRP) provides a formal mathematical model to this problem. This problems aims to find a subset of requirements or stakeholders that optimize a wanted attribute, such as profit or cost.
NRP can be reduced to the knapsack problem which, in the end, is a NP-Hard problem.

## Formal definition of NRP Mono-Objective

The problem can be addressed as a mixed integer linear programming (MILP) model. The next model assumes that the decision maker wants to maximize the profit subject to a cost restriction. 

- Let $$S$$ be the set of stakeholders with $|S| = n$
- Let $R$ be the set of requierements with $|R| = m$
- Let $ p $ be the max cost affordable.
- Let $ X = [x_1 , x_2 , … ,x_n ] $ be a binary array where the value of the cell $ i = 1$ if $i$ requirement if implemented. $ 0 $ otherwise 
- Let $ Y = [y_1, y_2,…,y_m] $ be a binary array where the value of the cell $ j = 1$ if the $j$ stakeholder is satisfied (that means, all of his requirement are implemented in the next release). $0$ otherwise.
- Let $ C = [c_1, c_2, …, c_n ] $ be an array of cost per requirement.
- Let $B = [b_1, b_2 ,… ,b_m] $ the profit of satisfy a stakeholder
- Let $P$ be the precedence relation between $(i,j)$ where $i,j$ are requirements; meaning that $i$ requirement must be selected if $j$ requirement is selected.
- Let $ I $ be the interest relation $(i,k)$ where $k$ stakeholder has interest over $i$ requierement.

With this parameters, the model looks like:

$$
max \ f(Y) = \sum_{i \in S} b_i \cdot y_i
$$
subject to
$$
\sum_{j \in R} c_j \cdot x_j \leq p
$$

$$
x_i \geq x_j \quad \forall (i,j) \in P
$$

$$
x_i \geq y_k \quad \forall (i,k) \in I
$$

$$
X \in \{0,1\}^n
$$
$$
Y \in \{0,1\}^n
$$