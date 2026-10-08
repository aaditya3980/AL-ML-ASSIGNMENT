import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Create a synthetic Multimodal dataset
# Modality 1: Text Data (Headlines)
# Modality 2: Numerical/Metadata (e.g., Shares count, Image trust score)
data = {
    "headline": [
        "Breaking: Aliens landed in New York today!",
        "Government announces new tax reform policy for next fiscal year.",
        "Miracle pill guarantees 100% weight loss in 2 days without diet!",
        "Scientists discover new species of deep sea coral off Pacific coast.",
        "You won't believe what this celebrity did at the red carpet event!",
        "Central Bank updates interest rates following quarterly review.",
        "Secret formula revealed to turn lead into pure gold instantly!",
        "Local university opens new research center dedicated to AI development.",
        "Shocking video shows ghost floating inside hospital!",
        "Ministry of Health releases updated vaccination guidelines.",
    ],
    "share_count": [
        15000,
        200,
        25000,
        150,
        18000,
        300,
        30000,
        100,
        22000,
        180,
    ],
    "image_trust_score": [0.1, 0.9, 0.2, 0.85, 0.3, 0.95, 0.15, 0.88, 0.1, 0.92],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # 1 = Fake News, 0 = Real News
}

df = pd.DataFrame(data)

# 2. Extract Text Features using TF-IDF (Text Modality)
tfidf = TfidfVectorizer(stop_words="english")
X_text = tfidf.fit_transform(df["headline"])

# 3. Extract and Scale Numerical Features (Metadata Modality)
scaler = StandardScaler()
X_num = scaler.fit_transform(df[["share_count", "image_trust_score"]])

# 4. Fuse both modalities (Multimodal Integration via Sparse Stacking)
X_multimodal = hstack([X_text, X_num])
y = df["label"]

# 5. Split into train/test sets using stratify to maintain balanced class distributions
X_train, X_test, y_train, y_test = train_test_split(
    X_multimodal, y, test_size=0.3, random_state=42, stratify=y
)

# 6. Train a Random Forest classifier on multimodal data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# 7. Make predictions and display performance metrics
y_pred = clf.predict(X_test)

print("=== Multimodal Fake News Detection Model Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Real News", "Fake News"]
    )
  }
