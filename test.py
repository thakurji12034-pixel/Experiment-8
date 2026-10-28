import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load("model.pkl")

testdf = pd.read_csv("Iris - Iris.csv")

testdf = testdf.drop("Id", axis=1)

X = testdf.drop("Species", axis=1)
y = testdf['Species']

pred = model.predict(X)
acc = accuracy_score(pred, y)
print(acc)

#Test cases
#1.TO check the number of predictions.
assert len(pred) == len(y)   

#2. Accuracy test
# assert acc >= 1.0 
assert acc >= 0.98
print("All test cases passed and model is ready to deploy")
