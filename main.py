import pandas as pd

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels (ham=0, spam=1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Convert text to numbers
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate model
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Test with custom message
msg = ["Congratulations! You have won a free ticket"]
msg_vec = vectorizer.transform(msg)

result = model.predict(msg_vec)

if result[0] == 1:
    print("Spam 🚨")
else:
    print("Not Spam ✅")