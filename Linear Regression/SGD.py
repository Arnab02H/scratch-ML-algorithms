### Stochastic Gradient Descent 
import numpy as np 
def stochastic_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000):
    """
    Stochastic Gradient Descent - Uses ONE data point at a time
    """
    print("Training with Stochastic Gradient Descent...")

    # Initialize parameters
    n_features = X.shape[1]
    weights = np.random.normal(0, 0.01, n_features)
    bias = 0
    loss_history = []

    m = X.shape[0]  # number of samples

    for epoch in range(max_iterations):
        # Shuffle data for each epoch
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0

        # Process one sample at a time
        for i in range(m):
            # Get single sample
            x_i = X_shuffled[i]
            y_i = y_shuffled[i]

            # Forward pass
            prediction = x_i @ weights + bias

            # Calculate loss for this sample
            sample_loss = (y_i - prediction) ** 2
            epoch_loss += sample_loss

            # Calculate gradients for this sample
            weight_gradient = -2 * x_i * (y_i - prediction)
            bias_gradient = -2 * (y_i - prediction)

            # Update parameters
            weights = weights - learning_rate * weight_gradient
            bias = bias - learning_rate * bias_gradient

        # Store average loss for this epoch
        avg_loss = epoch_loss / m
        loss_history.append(avg_loss)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    return weights, bias, loss_history
