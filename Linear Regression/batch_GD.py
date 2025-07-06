#### Batch Gradient Descent 
import numpy as np 
def batch_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000):
    """
    Batch Gradient Descent - Uses ALL data points in each iteration
    """
    print("TTRAINING LINEAR REG. WITH BATCH GRADIENT DESCENT----")

    #### Initialize parameters
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    bias = 0
    loss_history = []
    ### number of samples 
    m = X.shape[0]  

    for i in range(max_iterations):
      ### Forward Pass
        predictions = X @ weights + bias

        #### Calculate loss
        loss = np.mean((y - predictions) ** 2)
        loss_history.append(loss)
        ## Derivative w.r.to weight parameter
        weight_gradient = (-2/m) * X.T @ (y - predictions)
        ## Derivative w.r.to bias parameter
        bias_gradient = (-2/m) * np.sum(y - predictions)

        #### Update parameters using Gradient Descent Rule
        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient
        ### Print the loss after each 100 iteration
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.6f}")

    return weights, bias, loss_history
