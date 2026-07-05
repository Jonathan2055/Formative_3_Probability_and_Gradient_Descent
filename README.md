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

### How to Run

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

### Log-Likelihood

The log-likelihood measures how well the current model explains the data.

As the EM algorithm runs, the log-likelihood should increase (become less negative), indicating that the model is improving.

### Output

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

### Part One Conclusion

The EM algorithm successfully models the height data as a mixture of two Gaussian distributions. Unlike a simple mean-based split, EM uses probabilistic classification, making it more effective when the distributions overlap.

## Part 2: Bayesian Probability (IMDB Sentiment Analysis)

### Overview

This project applies **Bayes' Theorem** from scratch in Python to estimate how strongly certain keywords signal positive or negative sentiment in movie reviews.

Using the IMDB Movie Reviews Dataset, we selected keywords associated with positive and negative sentiment, then computed the probability that a review is positive given the presence of each keyword.

### How to Run

### Requirements
Install the required libraries:
Dependencies:
1. Pandas
2. Numpy

Import the dataset and update the path in this line:
```python
df = pd.read_csv("IMDB Dataset.csv")
```

Note: the dataset may throw a `ParserError` due to unescaped quote characters in review text. If this happens, load it with the Python engine instead:
```python
df = pd.read_csv("IMDB Dataset.csv", engine="python", on_bad_lines="skip")
```

### Run the Program
Run all cells from top to bottom.

The program will:

1. Load the review dataset.
2. Define selected positive and negative keywords.
3. Check keyword presence in individual reviews.
4. Compute the prior probability of a positive review.

### Part Two Conclusion
Bayesian probability provides a principled way to quantify how much a single keyword shifts our belief about a review's sentiment. Keywords that are common in positive reviews but rare overall produce the largest posterior shifts, confirming they are meaningful sentiment indicators — while generically common words fail to move the posterior far from the prior, showing they carry little diagnostic value on their own.