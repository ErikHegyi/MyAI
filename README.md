# MyAI
A personal project of mine, which I started, because I wanted to understand how artificial intelligence (machine and deep learning) actually works.  
The goal of this program is to build up a neural network from scratch.

To fully understand it, I started with the most basic machine learning model: a **single feature linear regression model**.  
After this, I want to add more models, and slowly work my way up to a neural network.

## Linear Regression
A linear regression model tries to find the linear function, which can describe the given data the best.  
A linear function looks like this: $y = ax + b$, where  
$y$ is the output of the function,\
$a$ is the **weight** of the function,\
$x$ is the input value of the function,\
$b$ is the bias of the function.

The model constantly shifts around the weight and the bias - the **parameters** until it minimizes the *loss* and the *cost*.  
The loss is the error for a singular piece of data:\
$L = y_{predicted} - y_{actual}$\
For example, if the value predicted by the model is 505, and the actual value is 500, then the loss is 5.  
The cost is the sum of the losses for each piece of training data that we have. It is calculated based on this formula:
$$C = \frac{1}{n} \sum_{i=0}^n L_i^2 = \frac{1}{n} \sum_{i=0}^n (\hat{Y_i} - Y_i)^2 = \frac{1}{n} \sum_{i=0}^n (ax_i + b - Y_i)^2$$\
where  
$n$ is the amount of data we have,  
$L_i = \hat{Y}_i-Y_i$ is the loss for the current piece of data, where $\hat{Y}$ is the vector of predicted values, and $Y$ is the vector of actual values,  
$a$ is the weight,  
$b$ is the bias,  
$x_i$ is the value that should be predicted.

The program constantly updates the parameters, until it finds the optimal values.
This is done by graphing the cost function, and looking for the lowest slope - this is called **gradient descent**.  
We derivate the parameters cost function with respect to parameters $a$ and $b$, and then iterate multiple times, each time updating the parameters.

```rust
fn train_model(iterations: u32, learning_rate: f64) {
    for _ in 0..iterations {
        weight -= learning_rate * weight_derivative;
        bias -= learning_rate * bias_derivative;
    }
}
```

## Polynomial Regression
Like linear regression, polynomial regression is a regression algorithm with a cost function and gradient descent.
The base formula for the linear function is: $y = ax + b$\
The base formula for the polynomial function is: $$y = b + \sum_{i=0}^n a_i x^{n-i}$$ or $$y = a_0x^n + a_1x^{n-1} + a_2x^{n-2} ... + a_{n - 1}x + n$$\
Where:\
$a$ is the vector of weights.\
$b$ is the bias.\
This allows for better accuracy if the data does not fit a linear function, but can still be described with a single line.

### The cost function
The cost function of a polynomial regression is the same function
$$C = \frac{1}{n} \sum_{i=0}^n L_i^2 = \frac{1}{n} \sum_{i=0}^n (\hat{Y_i} - Y_i)^2$$\
Except that now, the prediction will not be $a_ix_i + b$, but $b + \sum_{i=0}^n a_i x^{n - i}$\
where
$a_i$ is the weight (vector) for the current power,\
$x$ is the given feature vector\
$b$ is the bias.
### Ridge
### Lasso
### Elastic Net
## Logistic Regression
### Sigmoid
## Multinomial Logistic Regression
### Softmax
## Naive Bayes
### Gaussian Naive Bayes
### Multinomial Naive Bayes
### Bernoulli Naive Bayes
## Decision Tree
## Random Forests
## Support Vector Machines
### Linear Kernel
### Polynomial Kernel
### RBF Kernel
### Sigmoid Kernel
## Support Vector Regression
## K-Nearest Neighbors
## K-Nearest Neighbors Regression
## Ensemble Methods
### Bagging
### Boosting
### Voting
### Stacking
## Neural Networks
### ReLU
## Clustering
### K-Means
## Principal Component Analysis