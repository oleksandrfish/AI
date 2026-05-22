import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("poly_regression/fuel_consumption_vs_speed.csv")

X = df[["speed_kmh"]]
y = df["fuel_consumption_l_per_100km"]

best_degree = 0
best_mse = 999999
best_model = None

for degree in range(2, 6):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print(degree, mse, mae)
    if mse < best_mse:
        best_mse = mse
        best_degree = degree
        best_model = model

print("Best degree:", best_degree)

y_pred = best_model.predict(X)

print("MSE:", mean_squared_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))

new_speed = pd.DataFrame({"speed_kmh": [35, 95, 140]})
pred = best_model.predict(new_speed)

print("35:", pred[0])
print("95:", pred[1])
print("140:", pred[2])

plt.scatter(X, y, color="blue")
plt.plot(X, y_pred, color="red")
plt.show()
