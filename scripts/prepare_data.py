import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\AI-hyper\data\cst_data(pred) (1).csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("Temperature", axis=1)
y = df["Temperature"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\AI-hyper\data\X_train.csv", index=False)
X_test.to_csv(r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\AI-hyper\data\X_test.csv", index=False)
y_train.to_csv(r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\AI-hyper\data\y_train.csv", index=False)
y_test.to_csv(r"C:\Users\aksha\OneDrive\Desktop\PROJECTS\AI-hyper\data\y_test.csv", index=False)

print("✅ Data prepared successfully!")
