import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# 2. Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Initialize and train Naïve Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)
# 4. Make predictions on the test set
y_pred = model.predict(X_test)
# 5. Evaluate the model performance
print("=== Naïve Bayes Classifier Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# 6. Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
disp.plot(cmap='Blues')
plt.title("Naïve Bayes Confusion Matrix")
plt.show()
