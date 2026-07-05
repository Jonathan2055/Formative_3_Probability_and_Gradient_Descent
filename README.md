# Formative_3_Probability_and_Gradient_Descent
## Part 1 : Probability Distribution(Height Classification)

### Overview

This project implements the **Expectation-Maximization (EM) Algorithm** from scratch in Python.

The goal is to model height data as a mixture of two hidden Gaussian distributions which is father abd children. We assume that the dataset contains heights from two different populations (Children and Fathers), but we do not know which height belongs to which group.

Using the EM algorithm, we estimate:

* Mean (μ) of each group
* Variance (σ²) of each group
* Mixing coefficients (π)
* Posterior probabilities for classification

## How to Run

### Requirements
Install the required libraries:
This dependences are: 
1. Pandas
2. Numpy
Import the dataset and copy the path of the Dataset and paste it in this line ("df = pd.read_csv("/content/sample_data/GaltonFamilies.csv")
")

### Run the Program
run all cells from top to bottom.

The program will:

1. Load the height data.
2. Initialize Gaussian parameters.
3. Perform the Expectation (E) Step.
4. Perform the Maximization (M) Step.
5. Update parameters over multiple iterations.
6. Display optimization metrics.
7. Allow classification of a test height.

## Problem Statement

A simple approach would be to split the dataset at the global mean and classify all heights below the mean into one group and all heights above the mean into another.

This approach is not ideal because the two populations overlap. Heights close to the mean can easily be misclassified.

The EM algorithm solves this problem by assigning probabilities rather than making hard classifications.

## Expectation-Maximization (EM)

The EM algorithm repeatedly performs two steps:

### E-Step (Expectation)

For each height value:

* Calculate the probability of belonging to Children's Group.
* Calculate the probability of belonging to Father's Group.

These probabilities are called **responsibilities**.

### M-Step (Maximization)

Using the responsibilities:

* Update the means (μ₁ and μ₂).
* Update the variances (σ₁² and σ₂²).
* Update the mixing coefficients (π₁ and π₂).

The process repeats until the model converges.

## Log-Likelihood

The log-likelihood measures how well the current model explains the data.

As the EM algorithm runs, the log-likelihood should increase (become less negative), indicating that the model is improving.

# Output

The program prints:

* Iteration number
* Means (μ₁, μ₂)
* Variances (σ₁², σ₂²)
* Mixing coefficients (π₁, π₂)
* Log-Likelihood

It also accepts a test height and outputs the posterior probabilities of belonging to each group.

Example:

```text
Enter a height: 70

Probability Child: 27.06%
Probability Father: 72.94%
```

---

# Conclusion

The EM algorithm successfully models the height data as a mixture of two Gaussian distributions. Unlike a simple mean-based split, EM uses probabilistic classification, making it more effective when the distributions overlap.

