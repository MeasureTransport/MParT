---
title: 'MParT: Monotone Parameterization Toolkit'
tags:
  - Python
  - Julia
  - Matlab
  - c++
  - measure transport
  - Knothe-Rosenblatt rearrangement
  - isotonic regression
  - density estimation
  - Bayesian inference
authors:
  - name: Matthew Parno
    orcid: 0000-0002-9419-2693
    corresponding: true
    affiliation: 1
  - name: Paul-Baptiste Rubio
    orcid: 0000-0002-9765-1162
    affiliation: 2
  - name: Daniel Sharp
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Michael Brennan
    orcid: 0000-0000-0000-0000
    affiliation: 2 
  - name: Ricardo Baptista 
    orcid: 0000-0002-0421-890X
    affiliation: 2
  - name: Henning Bonart
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Youssef Marzouk 
    orcid: 0000-0000-0000-0000
    affiliation: 2
affiliations:
 - name: Dartmouth College, USA
   index: 1
 - name: Massachusetts Institute of Technology, USA
   index: 2
date: 22 July 2022
bibliography: paper.bib

---

# Summary

Transportation of measures is a rich area in applied mathematics that  constructs deterministic transformations--known as transport maps--which couple random variables [@santambrogio2015optimal]. These maps characterize a complex target distribution as a transformation of a simple reference distribution (e.g., a standard Gaussian). In the context of probabilistic modeling, transport maps permit easily generating samples from a target distribution and evaluating its probability density function. Monotone triangular maps are one class of transport maps that have several computational advantages over non-triangular maps. As a result, they are a core building block of normalizing flows architectures used in the machine learning community [@papamakarios2021normalizing]. Moreover, triangular maps are well suited to model conditional distributions, such as those appearing in Bayesian inference problem[@Marzouk2016]. The analytic form of transport maps for complex problems, however, are rarely known in closed-form. Hence, it is essential to develop tools for representing and estimating these maps given only limited information (e.g., samples from the target distribution or evaluations of its density up to a normalization constant). These tools should be efficient in order to use maps as part of online algorithms, as commonly found in sequential inference problems[@spantini2019coupling].

# Statement of need

The MParT package provides tools for parameterizing and using monotone maps. The aim of the package is to not sacrifice efficiency of evaluations that are needed to learn and evaluate such maps. The backend is in C++ and bindings are available to three high-level development languages: Julia, Matlab and Python. The map representation can be used in conjunction with optimization tools available in this language.

# Numerical results


# Acknowledgements

We acknowledge support from the US Office of Naval Research, US Department of Energy and Dartmouth College. 
