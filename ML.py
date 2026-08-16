import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load your CSV files
train_data = pd.read_csv("train_data - train_data.csv")
test_data = pd.read_csv("test_data - test_data.csv")

# 2. Automatically select the first column as feature (X) and second column as target (y)
X_train = train_data.iloc[:, [0]]
y_train = train_data.iloc[:, 1]

X_test = test_data.iloc[:, [0]]
y_test = test_data.iloc[:, 1]

# 3. Fit Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict on test dataset & calculate Mean Squared Error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")

# 5. Generate and display the plot
plt.figure(figsize=(8, 5))

# Scatter plot for actual test points
plt.scatter(
    X_test, y_test, color="blue", alpha=0.7, edgecolors="k", label="Test Data"
)

# Line plot for fitted model predictions
sort_idx = np.argsort(X_test.iloc[:, 0].values)
plt.plot(
    X_test.iloc[sort_idx, 0],
    y_pred[sort_idx],
    color="red",
    linewidth=2,
    label="Regression Line",
)

plt.title("Linear Regression Test Evaluation")
plt.xlabel(train_data.columns[0])
plt.ylabel(train_data.columns[1])
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save image and display
plt.savefig("regression_plot.png")
plt.show()