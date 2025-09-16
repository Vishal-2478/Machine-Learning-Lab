import numpy as np
from random import randint

# Generate sample data
np.random.seed(42)
n_samples = 1000
heights_cm = [randint(150, 200) for _ in range(n_samples)]
weights_kg = [randint(45, 120) for _ in range(n_samples)]

# Calculate BMI and obesity labels
heights_m = [h / 100 for h in heights_cm]
bmis = [weight / (height**2) for weight, height in zip(weights_kg, heights_m)]
is_obese = [1 if bmi >= 30 else 0 for bmi in bmis]


# Sigmoid function
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# Prepare data
X_height = np.array(heights_cm)
X_weight = np.array(weights_kg)
y = np.array(is_obese)

# Normalize features
height_mean, height_std = X_height.mean(), X_height.std()
weight_mean, weight_std = X_weight.mean(), X_weight.std()
X_height_norm = (X_height - height_mean) / height_std
X_weight_norm = (X_weight - weight_mean) / weight_std

# Create feature matrix with bias term
X = np.column_stack([np.ones(n_samples), X_height_norm, X_weight_norm])

# Initialize weights randomly
weights = np.random.normal(0, 0.01, 3)
learning_rate = 0.1
max_iterations = 1000
cost_history = []

# Training loop
for i in range(max_iterations):
    # Forward pass
    linear_pred = X.dot(weights)
    predictions = sigmoid(linear_pred)

    # Calculate cost
    predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
    cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    cost_history.append(cost)

    # Calculate gradients
    dw = (1 / n_samples) * X.T.dot(predictions - y)

    # Update weights
    weights -= learning_rate * dw

# Make predictions on training data
linear_pred = X.dot(weights)
train_probabilities = sigmoid(linear_pred)
train_predictions = (train_probabilities > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(train_predictions == y)

print(f"Training Accuracy: {accuracy:.3f}")
print(f"Predictions are: {np.unique(train_predictions)}")

# Show first 15 predictions
print("\nFirst 15 predictions:")
for i in range(15):
    print(
        f"Height: {heights_cm[i]} Weight: {weights_kg[i]} Actual: {y[i]} Predicted: {train_predictions[i]}"
    )


# Predict new people
def predict_new_person(height_cm, weight_kg):
    height_norm = (height_cm - height_mean) / height_std
    weight_norm = (weight_kg - weight_mean) / weight_std
    features = np.array([1, height_norm, weight_norm])
    linear_pred = features.dot(weights)
    probability = sigmoid(linear_pred)
    prediction = 1 if probability > 0.5 else 0
    return prediction


# Test with new people
test_people = [(170, 85), (160, 80), (180, 70), (150, 75), (190, 95)]
print("\nNew predictions:")
for height, weight in test_people:
    pred = predict_new_person(height, weight)
    print(f"Height: {height} Weight: {weight} Prediction: {pred}")
