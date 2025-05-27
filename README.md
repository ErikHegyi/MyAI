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
$x_i$ is the feature, that we are basing our prediction on.

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
The base formula for the polynomial function is: $$y = b + \sum_{i=0}^n a_i x^{n-i}$$ or $$y = a_0x^n + a_1x^{n-1} + a_2x^{n-2} ... + a_{n - 1}x + b$$\
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
Logistic regression is a **binary classification algorithm**, which uses the **sigmoid** function to map the output of a linear regression model into the $<0.0; 1.0>$ range.  
The closer the value is to 0.0, the more likely it is to be *A*, and the closer it is to 1.0, the more likely it is to *B*.
### Sigmoid
The sigmoid function maps all of our values between $0.0$ and $1.0$.
$$\sigma (x) = \frac{1}{1 + e^{-x}}$$
Where $e$ is Euler's number ($e \approx 2.718$).
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
K-Nearest Neighbors (**KNN**) maps the given data points on a 2-dimensional plane.  
After this, when receiving a new data point, it checks which existing data points are closer, and classifies it.  
Example:
> We mapped two bananas, and two apples. We give it another fruit. If it is closer to the apples, it classifies the new fruit as an apple, otherwise as a banana.

The formula for calculating the distance between two points is either the:
- **Euclidean distance** - $\sqrt{\sum_{i=1}^{d}(a_i - b_i)^2}$, where $d$ is the number of dimensions (for our two-dimensional plane $d = 2$ - x and y), and $a$ and $b$ our are points.
- **Manhattan distance** - $\sqrt{\sum_{i=1}^{d}|x_i - y_i|}$
- **Minkowski Distance** - $(\sum_{i=1}^n(x_i - y_i)^P)^{\frac{1}{P}}$

My model uses the **Euclidean distance**.

> The **k** in the K-Nearest neighbors stands for the **hyperparameter k**, which tells the model how many neighbors should it check.  
> For example, if $k = 2$, then it will only check the two closest neighbors.
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