---
title: "On Solver Optimization"
subtitle : "Measuring solving time of MILP solvers"
date: 2021-05-25T19:35:09-03:00
draft: true
linksAuthors : {"/members/leonardo-hoet" : "Leonardo Hoet"}
tags: ["example", "optimization"]
---


MILP models can take a lot of time to solve. We are trying to improve that.

<!--more-->

# Motivation 

Solving a MILP problem can be quite challenging, since algorithms used to do it have a close to exponential complexity. A [study](https://calhoun.nps.edu/handle/10945/29022) from Dr Smith under US. Naval Postgraduate School has shown that time complexity for branch and bound is 


$$
O(n^3 \cdot \int_{x}^{t} \gamma(x,t) \\ d\bold{S})
$$




In order to give a more concrete example, we have run an instance of next release problem in [GLPK](https://www.gnu.org/software/glpk/) and it took 43 hours to find the exact solution


```bash
➜  blog git:(master) ✗ time (echo && sleep 4)                

( echo && sleep 4; )  0,00s user 0,01s system 0% cpu 4,006 total
➜  blog git:(master) ✗  
```

As you can see, it is not good if you are ....


## Testing others solvers
We have testing various solvers...

### GLPK vs CBC

![GLPK vs CBC](/nrp_750c_3250r.png)
