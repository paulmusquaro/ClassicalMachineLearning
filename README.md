# ClassicalMachineLearning

This project demonstrates key methods in classical machine learning for predictive modeling, specifically applying linear regression. Linear regression is a foundational approach in machine learning, used to predict a target variable based on relationships with one or more input features. Here, we predict house prices based on features such as area, number of bathrooms, and number of bedrooms, utilizing gradient descent, analytical solutions, and `scikit-learn`.

## Library Used
This project uses a library essential for machine learning and scientific computation in Python:
+ `scikit-learn`: was initially developed by David Cournapeau as part of the Google Summer of Code project in 2007, with further contributions from the community and support from INRIA (French Institute for Research in Computer Science and Automation). Built on top of NumPy, SciPy, and matplotlib, scikit-learn provides efficient tools for data mining and machine learning, including regression, classification, clustering, and dimensionality reduction. Since its inception, scikit-learn has become one of the most widely used libraries in machine learning for Python, thanks to its simplicity, efficiency, and extensive functionality.


## Project Overview

The project consists of three key components, illustrating different methods for parameter estimation and model evaluation in linear regression:

1. **Gradient Descent Solution**:
    + Implements linear regression using gradient descent to iteratively minimize the loss function and find optimal parameters.
    + Functions:
        + Hypothesis Function: Computes the linear predictions based on input features.
        + Loss Function: Calculates mean squared error (MSE) to assess model performance.
        + Gradient Descent Step: Updates parameters using gradients to minimize the loss iteratively.
    + Output: Shows the gradual reduction of the loss function and the optimal parameters (`w`) for the model.

2. **Analytical Solution**:
    + Calculates the optimal parameters directly using the closed-form solution of linear regression, or the Normal Equation.
    + This approach finds the exact solution for the weights without iterative approximation.
    + Output: Optimal parameters for the linear model.

3. **Verification with `scikit-learn`**:
    + Uses `LinearRegression` from `scikit-learn` to build a regression model for validation.
    + Compares predictions from the gradient descent and analytical methods to those from the `scikit-learn` model.


## Conda (Setup and Environment)

To make the project reproducible and ensure smooth package management, this project uses Conda as a package and environment manager. Below are the steps to set up the environment:


1. **Install Conda**:
If you haven't installed Conda yet, you can download it from the official [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) websites. Anaconda is a larger distribution with more pre-installed packages, while Miniconda is a smaller, minimal version. Choose whichever suits your needs.

2. **Create a new environment:** Open your terminal and run the following command to create a new Conda environment with Python 3.12:

    ```bash
    conda create --name new_conda_env python=3.12
    ```

3. **Activate the environment:** Once the environment is created, activate it by running:

    ```bash
    conda activate new_conda_env
    ```

4. **Install required packages (Jupyter, NumPy, MatPlotLib, Pandas and Scikit-Learn)**

    ```bash
    conda install jupyter numpy matplotlib pandas scikit-learn
    ```

5. **Run Jupyter Notebook**

    ```bash
    jupyter notebook
    ```

***
## Conclusion
This project demonstrates fundamental concepts in linear regression and parameter estimation using both iterative (gradient descent) and direct (analytical) methods. By comparing these techniques and validating them against `scikit-learn`, this project offers an introductory look into classical machine learning techniques for predictive modeling, providing a foundation for more complex models and methods.