# MyAI
A personal project of mine, which I started, because I wanted to understand how artificial intelligence (machine and deep learning) actually works.  
The goal of this program is to build up a neural network from scratch.

To fully understand it, I started with the most basic machine learning model: a **single feature linear regression model**.  
After this, I want to add more models, and slowly work my way up to a neural network.

## Linear Regression
A linear regression model tries to find the linear function, which can describe the given data the best.  
A linear function looks like this: $y = ax + b$, where  
$y$ is the output of the function,
$a$ is the **weight** of the function,  
$x$ is the input value of the function,  
$b$ is the bias of the function.

The model constantly shifts around the weight and the bias - the **parameters** until it minimizes the *loss* and the *cost*.  
The loss is the error for a singular piece of data:  
for example, if the value predicted by the model is 505, and the actual value is 5, then the loss is 5.  
The cost is the sum of the losses for each piece of training data that we have. It is calculated based on this formula:
$$C = \frac{1}{n} \sum_{i=0}^n L_i^2$$  
$$C = \frac{1}{n} \sum_{i=0}^n ( \hat{Y}_i-Y_i)^2$$  
$$C = \frac{1}{n} \sum_{i=0}^n (ax_i+b-Y_i)^2$$  
$$C = \frac{1}{n} \sum_{i=0}^n L_i^2 = \frac{1}{n} \sum_{i=0}^n \hat{Y}_i - Y_i = \frac{1}{n} \sum_{i=0}^n ax_i + b - Y_i$$
where  
$n$ is the amount of data we have,  
$L_i = \hat{Y}_i-Y_i$ is the loss for the current piece of data, where $\hat{Y}$ is the vector of predicted values, and $Y$ is the vector of actual values,  
$a$ is the weight,  
$b$ is the bias,  
$x_i$ is the value that should be predicted.

The program constantly updates the parameters, until it finds the optimal values.
This is done by looking for the graphing the cost function, and looking for the lowest slope - this is called **gradient descent**.  
We derivate the parameters cost function with respect to parameters $a$ and $b$, and then iterate multiple times, each time updating the parameters.

```rust
fn train_model(iterations: u32, learning_rate: f64) {
    for _ in 0..iterations {
        weight -= learning_rate * weight_derivative;
        bias -= learning_rate * bias_derivative;
    }
}
```