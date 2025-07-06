import numpy as np 
def create_sample_data(n_samples=1000, n_features=3):
    print(f"Creating sample data with {n_samples} samples and {n_features} features...")

    np.random.seed(42)

    # Generate random features
    X = np.random.normal(0, 1, (n_samples, n_features))

    # Create true weights and bias
    true_weights = np.array([1.5, -2.0, 0.5])[:n_features]
    true_bias = 0.5

    # Generate target values with some noise
    y = X @ true_weights + true_bias + np.random.normal(0, 0.1, n_samples)

    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")

    return X, y, true_weights, true_bias

### STEP 6: NORMALIZE DATA (IMPORTANT FOR GRADIENT DESCENT)

def normalize_data(X_train, X_test):
    """Normalize features to have mean=0 and std=1"""
    print("Normalizing data...")

    # Calculate mean and std from training data
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # Normalize both training and test data
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    return X_train_norm, X_test_norm

