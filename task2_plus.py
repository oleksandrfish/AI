import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv("dz_regression/energy_usage_plus.csv")

X = df[["temperature", "humidity", "season", "hour", "district_type", "is_weekend"]]
y = df["consumption"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["season", "district_type"])
    ],
    remainder="passthrough"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("prep", preprocessor),
    ("model", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.score(X_test, y_test))
print(mean_absolute_percentage_error(y_test, y_pred) * 100)

sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()
