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

## Part 3: Gradient Descent (Manual Calculation)

### Overview

This part demonstrates the **manual implementation of Gradient Descent** for a simple linear regression model using matrix multiplication.

The objective is to understand how Gradient Descent iteratively updates the model parameters to minimize the **Mean Squared Error (MSE)** between the predicted values and the actual target values.

Unlike the coded implementation in Part 4, every calculation in this section is performed manually to illustrate the mathematical process behind parameter optimization.

### Given Parameters

The calculations use the following initial values:

```text
m = [-1, 2]

b = [1, 1]

X = [[1, 3],
     [4,10]]

y = [5, 6]
```

Learning Rate:

```text
α = 0.01
```

### Manual Procedure

For each iteration, the following steps are performed:

1. Compute the predicted values using:

```math
\hat{y}=Xm+b
```

2. Compute the error vector:

```math
e=\hat{y}-y
```

3. Compute the Mean Squared Error (MSE):

```math
MSE=\frac{1}{n}\sum(\hat{y}-y)^2
```

4. Compute the gradients using matrix multiplication:

```math
\frac{\partial MSE}{\partial m}=\frac{2}{n}X^Te
```

```math
\frac{\partial MSE}{\partial b}=\frac{2}{n}e
```

5. Update the parameters using Gradient Descent:

```math
m_{new}=m_{old}-\alpha\frac{\partial MSE}{\partial m}
```

```math
b_{new}=b_{old}-\alpha\frac{\partial MSE}{\partial b}
```

6. Repeat the process for four iterations.

### Results

Over the four iterations, the model parameters gradually moved toward values that reduced the prediction error.

| Iteration | MSE |
|-----------|------:|
| Initial | 61.0000 |
| 1 | 6.5033 |
| 2 | 2.4974 |
| 3 | 2.1644 |
| 4 | 2.0994 |

The continuous reduction in MSE demonstrates that Gradient Descent successfully optimized the model parameters by moving in the direction of the steepest decrease of the cost function.

### Part Three Conclusion

The manual calculations provided a step-by-step demonstration of how Gradient Descent learns from data. By repeatedly computing predictions, evaluating the error, calculating gradients, and updating the parameters, the model progressively reduced the Mean Squared Error. This illustrates the fundamental optimization process used in machine learning algorithms before implementing the same procedure in code.

## Part 4: Gradient Descent in Code

### Overview

This section implements the **Gradient Descent Algorithm** for a multiple linear regression model using matrix multiplication. The objective is to iteratively update the model parameters (**m** and **b**) to minimize the **Mean Squared Error (MSE)** between the predicted and actual values.

SciPy is used to numerically compute the derivatives of the cost function, while Matplotlib is used to visualize how the parameters and error change over successive iterations.

### How to Run

### Requirements

Install the required libraries:

Dependencies:

1. NumPy
2. SciPy
3. Matplotlib

### Run the Program

Run all notebook cells from top to bottom.

The program will:

1. Initialize the feature matrix, target values, and model parameters.
2. Compute predictions using matrix multiplication.
3. Calculate the Mean Squared Error (MSE).
4. Compute the numerical derivatives of the cost function using SciPy.
5. Update the parameters (**m** and **b**) using Gradient Descent.
6. Repeat the update process for three iterations.
7. Compute the final predictions using the optimized parameters.
8. Display plots showing parameter updates and error reduction over the iterations.

### Gradient Descent

Gradient Descent is an optimization algorithm that minimizes a cost function by repeatedly updating the model parameters in the direction opposite to the gradient.

For each iteration, the algorithm:

- Computes the current predictions.
- Calculates the Mean Squared Error (MSE).
- Estimates the gradient of the cost function using SciPy.
- Updates the values of **m** and **b**.
- Stores the updated parameters and error for visualization.

### Mean Squared Error (MSE)

The Mean Squared Error measures the average squared difference between the predicted values and the actual values.

As Gradient Descent progresses, the MSE should decrease, indicating that the model is learning parameters that better fit the data.

### Output

The program displays:

- Current values of **m** and **b** for each iteration.
- Gradients of **m** and **b**.
- Updated parameter values after each iteration.
- Final predictions.
- Final Mean Squared Error (MSE).

It also generates two plots:

1. A graph showing how **m** and **b** change over each iteration.
2. A graph showing how the Mean Squared Error decreases over the iterations.


### Part Four Conclusion

The Gradient Descent implementation successfully minimizes the Mean Squared Error by iteratively updating the model parameters using numerical derivatives computed with SciPy. The parameter values gradually converge while the error decreases across iterations, demonstrating that the algorithm is moving toward a better-fitting linear regression model.