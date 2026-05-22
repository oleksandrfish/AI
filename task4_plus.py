import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("dz_logistic/internship_candidates_cefr_final.csv")

levels = {
    "Elementary": 1,
    "Pre-Intermediate": 2,
    "Intermediate": 3,
    "Upper-Intermediate": 4,
    "Advanced": 5
}

df["EnglishLevel"] = df["EnglishLevel"].map(levels)

X = df[["Experience", "Grade", "EnglishLevel", "Age", "EntryTestScore"]]
y = df["Accepted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

df["Probability"] = model.predict_proba(X)[:, 1]

plt.scatter(df["EntryTestScore"], df["EnglishLevel"], c=df["Probability"], cmap="plasma")
plt.xlabel("EntryTestScore")
plt.ylabel("EnglishLevel")
plt.title("Probability of Acceptance")
plt.colorbar()
plt.show()
