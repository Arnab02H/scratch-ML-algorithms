import numpy as np 
def mini_batch_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, batch_size=32):
    """
    Mini-batch Gradient Descent - Uses small batches of data
    """
    print(f"Training with Mini-batch Gradient Descent (batch size: {batch_size})...")

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
        num_batches = 0

        # Process data in mini-batches
        for i in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = X_batch.shape[0]

            # Forward pass
            predictions = X_batch @ weights + bias

            # Calculate loss for this batch
            batch_loss = np.mean((y_batch - predictions) ** 2)
            epoch_loss += batch_loss
            num_batches += 1

            # Calculate gradients for this batch
            weight_gradient = (-2/batch_m) * X_batch.T @ (y_batch - predictions)
            bias_gradient = (-2/batch_m) * np.sum(y_batch - predictions)

            # Update parameters
            weights = weights - learning_rate * weight_gradient
            bias = bias - learning_rate * bias_gradient

        # Store average loss for this epoch
        avg_loss = epoch_loss / num_batches
        loss_history.append(avg_loss)

        # Print progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    return weights, bias, loss_history
