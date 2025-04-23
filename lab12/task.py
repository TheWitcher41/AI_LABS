import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('loan_dataset.csv')

# Extract relevant features and target variable
X = data[['income', 'credit_score']].values
y = data['loan_status'].apply(lambda x: 1 if x == 'Approved' else 0).values

# Initialize weights and bias with specified values
weights = np.array([0.2, 0.5])
bias = 0.08
learning_rate = 0.01

def sigmoid(z):
    # Clip input values to avoid overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

def binary_cross_entropy(y_true, y_pred):
    # Clip predicted probabilities to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Manual calculations for one epoch
# Predict probabilities
y_pred = predict(X, weights, bias)

# Calculate the error (loss)
loss = binary_cross_entropy(y, y_pred)
print(f'Initial loss: {loss}')

# Calculate gradients
dw = np.dot(X.T, (y_pred - y)) / y.size
db = np.sum(y_pred - y) / y.size

# Update weights and bias
weights -= learning_rate * dw
bias -= learning_rate * db

# Predict probabilities after weight update
y_pred = predict(X, weights, bias)

# Calculate the error (loss) after weight update
loss = binary_cross_entropy(y, y_pred)
print(f'Loss after one epoch: {loss}')

# Train the model programmatically for 10 epochs
for epoch in range(10):
    # Predict probabilities
    y_pred = predict(X, weights, bias)
    
    # Calculate gradients
    dw = np.dot(X.T, (y_pred - y)) / y.size
    db = np.sum(y_pred - y) / y.size
    
    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    # Calculate the error (loss)
    loss = binary_cross_entropy(y, y_pred)
    print(f'Epoch {epoch + 1}, Loss: {loss}')

# Recalculate the error using the final weights
# Predict probabilities with final weights
y_pred = predict(X, weights, bias)

# Calculate the final error (loss)
final_loss = binary_cross_entropy(y, y_pred)
print(f'Final loss after 10 epochs: {final_loss}')

# Show final weights
print(f'Final weights: {weights}')
print(f'Final bias: {bias}')