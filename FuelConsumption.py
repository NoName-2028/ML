import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load the dataset
df = pd.read_csv("fuel_consumption_dataset - fuel_consumption_dataset.csv")

# 2. Select numeric feature (X) and target variable (y)
X = df[["ENGINESIZE"]]  # Feature: Engine Size [Using two third bracket for defining it as a 2D dataset]
y = df["CO2EMISSIONS"]  # Target: CO2 Emissions [predict or test data]

# 3. Split into 70% Training data and 30% Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict on Test set and compute MSE
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Test Mean Squared Error (MSE): {mse:.4f}")

# 6. Plot the results
plt.figure(figsize=(8, 5))

# Scatter plot for test points
plt.scatter(
    X_test,
    y_test,
    color="blue",
    alpha=0.5,
    edgecolors="k",
    label="Actual Test Data (30%)",
)

# Regression line
plt.plot(
    X_test, y_pred, color="red", linewidth=2, label="Linear Regression Line"
)

plt.title("Engine Size vs CO2 Emissions (70:30 Split)")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save image and show
plt.savefig("fuel_consumption_plot.png")
plt.show()
