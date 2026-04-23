import nbformat as nbf

nb = nbf.v4.new_notebook()

# Add cells
cells = []

# Title
cells.append(nbf.v4.new_markdown_cell("# Ensemble Learning Techniques: Stock Movement Prediction\n\nThis notebook demonstrates Ensemble Learning using Random Forest for Stock Prediction. We will use historical stock data to predict whether the stock price will go UP or DOWN the following day."))

# Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib"""))

# Download Data
cells.append(nbf.v4.new_markdown_cell("## 1. Download and Prepare Dataset\n\nWe will download Apple (AAPL) stock data using `yfinance`."))
cells.append(nbf.v4.new_code_cell("""# Download data
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2025-01-01')

# Flatten columns if multi-index
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join(col).strip() for col in data.columns.values]

# Calculate features
data['Returns'] = data['Close_AAPL'].pct_change()
data['MA_10'] = data['Close_AAPL'].rolling(window=10).mean()
data['MA_50'] = data['Close_AAPL'].rolling(window=50).mean()

# Target Variable: 1 if tomorrow's price is higher than today's, else 0
data['Target'] = np.where(data['Close_AAPL'].shift(-1) > data['Close_AAPL'], 1, 0)

data = data.dropna()

# Save the dataset
data.to_csv('stock_dataset.csv')
data.head()"""))

# Train Test Split
cells.append(nbf.v4.new_markdown_cell("## 2. Train-Test Split\n\nWe'll use standard features (Returns, Moving Averages) to predict the Target."))
cells.append(nbf.v4.new_code_cell("""features = ['Close_AAPL', 'Volume_AAPL', 'Returns', 'MA_10', 'MA_50']
X = data[features]
y = data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
print("Training set size:", len(X_train))
print("Test set size:", len(X_test))"""))

# Train Random Forest
cells.append(nbf.v4.new_markdown_cell("## 3. Train Random Forest Model\n\nRandom Forest is an ensemble learning method based on bagging decision trees."))
cells.append(nbf.v4.new_code_cell("""rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

# Predictions
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.4f}")
print("Classification Report:\\n", classification_report(y_test, rf_preds))"""))

# Compare with Logistic Regression
cells.append(nbf.v4.new_markdown_cell("## 4. Compare with Logistic Regression\n\nTo show the advantage of ensemble techniques, let's compare with a simpler baseline model."))
cells.append(nbf.v4.new_code_cell("""lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")"""))

# Evaluate
cells.append(nbf.v4.new_markdown_cell("## 5. Evaluate Performance\n\nLet's visualize the confusion matrix for Random Forest."))
cells.append(nbf.v4.new_code_cell("""cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()"""))

# Save Model
cells.append(nbf.v4.new_markdown_cell("## 6. Save the Trained Model\n\nWe will save the model using `joblib` so it can be deployed via Streamlit."))
cells.append(nbf.v4.new_code_cell("""joblib.dump(rf_model, 'model.pkl')
print("Model saved as model.pkl")"""))

nb['cells'] = cells

with open('Ensemble_Learning_Techniques.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
