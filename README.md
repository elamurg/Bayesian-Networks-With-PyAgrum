## Lab 2 Exercises: Bayesian Inference with pyAgrum

This repository contains Python implementations for solving probability problems using Bayesian Networks and Bayes' Theorem, based on the **Risk and Decision-Making for Machine Learning** lab.

---

## Question 2: Pregnancy Test Probability

This exercise calculates the posterior probability of a pregnancy given a positive test result using Bayes' Theorem.

### Problem Overview

* 
**Prior Probability**: .


* 
**True Positive Rate**: .


* 
**False Positive Rate**: .



### Implementation (`question2.py`)

The script uses `pyagrum` to model the causal relationship where the state of being pregnant influences the test outcome.

* 
**Manual Calculation**: Result is approximately **0.9252**.


* **PyAgrum Verification**: Confirms the manual result through exact inference using `LazyPropagation`.

---

## Question 3: Car Electrical System

This exercise models a more complex system where multiple components (Battery, Gas, Ignition) interact to determine if a car will start.

### Causal Structure

The network is built based on the following dependencies:

* **Battery** is a parent to **Radio**, **Lights**, and **Ignition**.
* **Ignition** and **Gas** are parents to **Engine Starts**.

### Key Inferences

The implementation calculates:

1. 
**P(EngineStarts | Lights=True)**: The probability the car starts given that we observe the lights are functional.


2. 
**P(EngineStarts | Lights=True, Gas=True)**: The probability the car starts when we know both the lights are on and there is sufficient fuel.



### Node Probability Tables (NPTs)

The model incorporates specific failure rates, such as:

* A **10%** chance the battery is dead.


* A **25%** chance the car is out of gas.


* A **5%** chance the engine fails to start even if both Ignition and Gas are functional.



---

## Requirements

* **Python 3.x**
* **pyagrum** library

To install the necessary library, run:

```bash
pip install pyagrum

```

## How to Run

Execute the scripts directly from your terminal:

```bash
python question2.py
python question3.py

```

**Would you like me to add a section to this README explaining how to interpret the results of the "Engine Starts" query?**