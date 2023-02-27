# Statistical-Learning-Methods
What I have learned from the course "Statistical Learning Methods" in Warsaw School of Economics.

### Plotting simulated R-squared
File name: "plotting_simulated_R-squared.py" <br />
A homework for the very beginning of the course in which I had to write a Python code to plot a simulated R-squared at each sample size.

---
$$R^{2} = 1- \dfrac{\sum(Y_{i}-\hat{Y})^{2}}{\sum{(Y_{i}-\overline{Y})^{2}}} = 1 - \dfrac{MSE}{Var(Y)} \tag{1}$$

$Y$: response variable to be predicted by model <br />
$\hat{Y}$: predicted value for $Y$ by model <br />
$\overline{Y}$: average of $Y$ over $n$ observations <br />
$MSE$: mean of squared error, $SSE/n$ <br />
$Var(Y)$: variance of Y 

**R-squared from a simple linear model**

$$Y = 1 + X + e \tag{2}$$

$Y$: response variable <br />
$X$: independent variable following a standard Gaussian, $N(0, 1)$ <br />
$e$: error term following a standard Gaussain, $N(0, 1)$


![images/simple_regression_r2_convergence](images/simple_regression_r2_convergence.png)


