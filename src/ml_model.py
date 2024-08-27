import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
data = pd.read_csv('../data/dataset_1.csv')

# Data Preprocessing
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save Model
joblib.dump(model, '../models/ml_model.pkl')
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
