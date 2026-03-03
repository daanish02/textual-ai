# Classical Text Classification

## Table of Contents

- [Introduction](#introduction)
- [Classification Fundamentals](#classification-fundamentals)
- [Feature Engineering for Text](#feature-engineering-for-text)
- [Naive Bayes Classifier](#naive-bayes-classifier)
- [Logistic Regression](#logistic-regression)
- [Support Vector Machines (SVM)](#support-vector-machines-svm)
- [Decision Trees and Random Forests](#decision-trees-and-random-forests)
- [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
- [Model Evaluation](#model-evaluation)
- [Handling Imbalanced Data](#handling-imbalanced-data)
- [Multi-Class and Multi-Label Classification](#multi-class-and-multi-label-classification)
- [Practical Applications](#practical-applications)
- [Best Practices](#best-practices)
- [Summary](#summary)
- [Next Steps](#next-steps)

## Introduction

Text classification assigns predefined categories to text documents. It's one of the most fundamental NLP tasks with wide-ranging applications.

**Common applications**:

- **Sentiment analysis**: Positive/negative/neutral
- **Spam detection**: Spam/not spam
- **Topic categorization**: Sports, politics, technology, etc.
- **Intent classification**: User query understanding
- **Language identification**: Detecting document language

**Classification pipeline**:

```
Text Document
    ↓
Preprocessing (tokenization, cleaning)
    ↓
Feature Extraction (BoW, TF-IDF, n-grams)
    ↓
Classifier (Naive Bayes, SVM, etc.)
    ↓
Predicted Class
```

This guide covers classical machine learning approaches before the deep learning era.

## Classification Fundamentals

### Supervised Learning

Text classification is **supervised learning**: learn from labeled examples.

**Training data**:

```
[
    ("I love this product!", "positive"),
    ("Terrible experience", "negative"),
    ("Great quality", "positive"),
    ("Waste of money", "negative"),
    ...
]
```

**Goal**: Learn function $f: X \rightarrow Y$

- $X$ = feature space (text representation)
- $Y$ = label space (categories)

### Classification Types

**Binary classification**: Two classes

```python
# Spam detection
y = {0, 1}  # 0 = not spam, 1 = spam
```

**Multi-class classification**: Multiple mutually exclusive classes

```python
# Topic classification
y = {sports, politics, technology, entertainment}
```

**Multi-label classification**: Multiple non-exclusive labels

```python
# Movie genres
y = {action, comedy, drama}  # Can be [action, comedy], [drama], etc.
```

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

# Example data
texts = [
    "I love machine learning",
    "Deep learning is amazing",
    "This movie was terrible",
    "Worst experience ever",
    "Great product, highly recommend"
]
labels = [1, 1, 0, 0, 1]  # 1 = positive, 0 = negative

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# 5-fold cross-validation
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

## Feature Engineering for Text

### Bag-of-Words Features

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are pets"
]
labels = [0, 0, 1]

# Create BoW features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

print(f"Feature matrix shape: {X.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Features: {vectorizer.get_feature_names_out()}")
```

### TF-IDF Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF features
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
X_tfidf = tfidf.fit_transform(texts)

print(f"TF-IDF matrix shape: {X_tfidf.shape}")
```

### N-gram Features

```python
# Unigrams + bigrams
vectorizer = CountVectorizer(ngram_range=(1, 2))

texts = [
    "machine learning is great",
    "deep learning is powerful"
]

X = vectorizer.fit_transform(texts)
features = vectorizer.get_feature_names_out()

print("Features include:")
for feat in features[:10]:
    print(f"  {feat}")
```

### Character N-grams

```python
# Character-level features (useful for language detection, misspellings)
char_vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 4)
)

texts = ["hello world", "helo world"]  # Second has typo
X_char = char_vectorizer.fit_transform(texts)

print("Character n-grams capture spelling variations")
```

### Custom Features

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TextStatistics(BaseEstimator, TransformerMixin):
    """Extract statistical features from text."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Extract features for each document."""
        features = []

        for text in X:
            # Basic statistics
            num_words = len(text.split())
            num_chars = len(text)
            avg_word_length = num_chars / num_words if num_words > 0 else 0
            num_uppercase = sum(1 for c in text if c.isupper())
            num_punctuation = sum(1 for c in text if c in '!?.,;:')

            features.append([
                num_words,
                num_chars,
                avg_word_length,
                num_uppercase,
                num_punctuation
            ])

        return np.array(features)

# Example
texts = [
    "I LOVE THIS!!!",
    "this is okay"
]

stat_transformer = TextStatistics()
stats = stat_transformer.transform(texts)

print("Statistical features:")
print(stats)
```

### Feature Union (Combining Features)

```python
from sklearn.pipeline import FeatureUnion

# Combine TF-IDF with statistical features
feature_union = FeatureUnion([
    ('tfidf', TfidfVectorizer(max_features=100)),
    ('stats', TextStatistics())
])

X_combined = feature_union.fit_transform(texts)

print(f"Combined feature shape: {X_combined.shape}")
# Shape: (n_samples, tfidf_features + stat_features)
```

## Naive Bayes Classifier

### Probability Basics

Naive Bayes applies Bayes' theorem with "naive" independence assumption:

$$P(C|D) = \frac{P(D|C) \cdot P(C)}{P(D)}$$

Where:

- $C$ = class
- $D$ = document (features)
- $P(C|D)$ = posterior probability
- $P(D|C)$ = likelihood
- $P(C)$ = prior probability

**Naive assumption**: Features are independent given class

$$P(D|C) = P(w_1, w_2, ..., w_n | C) = \prod_{i=1}^{n} P(w_i | C)$$

### Multinomial Naive Bayes

Best for text classification with count features.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# Sample data
texts_train = [
    "free money win prize",
    "hello friend how are you",
    "claim your prize now",
    "meeting scheduled for tomorrow",
    "win lottery click here",
    "project deadline next week"
]
labels_train = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']

texts_test = [
    "congratulations you won",
    "let's meet for lunch"
]
labels_test = ['spam', 'ham']

# Vectorize
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts_train)
X_test = vectorizer.transform(texts_test)

# Train Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, labels_train)

# Predict
predictions = nb_classifier.predict(X_test)
probabilities = nb_classifier.predict_proba(X_test)

print("Predictions:")
for text, pred, probs in zip(texts_test, predictions, probabilities):
    print(f"  Text: '{text}'")
    print(f"  Prediction: {pred}")
    print(f"  Probabilities: spam={probs[0]:.3f}, ham={probs[1]:.3f}")
    print()
```

### Understanding Naive Bayes Parameters

```python
# Alpha (smoothing parameter)
# Higher alpha = more smoothing = less overfitting

nb_low_smooth = MultinomialNB(alpha=0.1)
nb_high_smooth = MultinomialNB(alpha=10.0)

nb_low_smooth.fit(X_train, labels_train)
nb_high_smooth.fit(X_train, labels_train)

print(f"Low smoothing accuracy: {nb_low_smooth.score(X_test, labels_test):.3f}")
print(f"High smoothing accuracy: {nb_high_smooth.score(X_test, labels_test):.3f}")
```

### Feature Importance in Naive Bayes

```python
def show_most_informative_features(vectorizer, classifier, n=10):
    """Show most important features for each class."""
    feature_names = vectorizer.get_feature_names_out()

    for class_idx, class_name in enumerate(classifier.classes_):
        # Get log probabilities for this class
        log_probs = classifier.feature_log_prob_[class_idx]

        # Get top features
        top_indices = np.argsort(log_probs)[-n:][::-1]

        print(f"\nTop features for class '{class_name}':")
        for idx in top_indices:
            word = feature_names[idx]
            prob = np.exp(log_probs[idx])
            print(f"  {word:15}: {prob:.4f}")

show_most_informative_features(vectorizer, nb_classifier)
```

### Bernoulli Naive Bayes

For binary (presence/absence) features.

```python
from sklearn.naive_bayes import BernoulliNB

# Binary features (word present or not)
vectorizer_binary = CountVectorizer(binary=True)
X_train_binary = vectorizer_binary.fit_transform(texts_train)
X_test_binary = vectorizer_binary.transform(texts_test)

# Train Bernoulli NB
bnb = BernoulliNB()
bnb.fit(X_train_binary, labels_train)

print(f"Bernoulli NB accuracy: {bnb.score(X_test_binary, labels_test):.3f}")
```

## Logistic Regression

### Linear Classification

Logistic regression models probability using sigmoid function:

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$

**Decision boundary**: Linear in feature space

### Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Prepare data
texts = [
    "I love this product amazing quality",
    "terrible waste of money",
    "great experience highly recommend",
    "worst purchase ever made",
    "excellent service fast shipping",
    "disappointed poor quality",
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train logistic regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X, labels)

# Test
test_texts = [
    "amazing product love it",
    "terrible experience"
]
X_test = vectorizer.transform(test_texts)
predictions = lr.predict(X_test)
probabilities = lr.predict_proba(X_test)

print("Logistic Regression Predictions:")
for text, pred, prob in zip(test_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    print(f"  '{text}'")
    print(f"  Prediction: {sentiment} (confidence: {confidence:.3f})")
    print()
```

### Regularization

```python
# L2 regularization (Ridge)
lr_l2 = LogisticRegression(penalty='l2', C=1.0)

# L1 regularization (Lasso) - feature selection
lr_l1 = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')

# C is inverse of regularization strength
# Smaller C = stronger regularization

for C in [0.01, 0.1, 1.0, 10.0]:
    lr = LogisticRegression(C=C, random_state=42)
    lr.fit(X, labels)
    print(f"C={C:5.2f}: Training accuracy = {lr.score(X, labels):.3f}")
```

### Feature Weights

```python
def show_feature_weights(vectorizer, classifier, n=10):
    """Show most important features for logistic regression."""
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    # Most positive (indicating class 1)
    top_positive_idx = np.argsort(coefs)[-n:][::-1]
    print("Top features for POSITIVE class:")
    for idx in top_positive_idx:
        print(f"  {feature_names[idx]:15}: {coefs[idx]:+.4f}")

    # Most negative (indicating class 0)
    top_negative_idx = np.argsort(coefs)[:n]
    print("\nTop features for NEGATIVE class:")
    for idx in top_negative_idx:
        print(f"  {feature_names[idx]:15}: {coefs[idx]:+.4f}")

show_feature_weights(vectorizer, lr)
```

## Support Vector Machines (SVM)

### Concept

SVM finds the hyperplane that maximizes the margin between classes.

```
Visual (2D):

  Class 0    |    Class 1
             |
    o        |        x
   o o       |       x x
    o        |        x
             |
    ← margin → margin →
             ↑
       decision boundary
```

**Key idea**: Maximize distance to nearest points (support vectors)

### Linear SVM

```python
from sklearn.svm import LinearSVC

# Prepare data
texts = [
    "love this amazing great",
    "hate terrible awful",
    "excellent fantastic wonderful",
    "disappointing poor bad",
    "outstanding brilliant superb",
    "worst horrible disgusting"
]
labels = [1, 0, 1, 0, 1, 0]

# Vectorize with TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train LinearSVC (faster than SVC for linear kernel)
svm = LinearSVC(random_state=42, max_iter=10000)
svm.fit(X, labels)

# Evaluate
test_texts = ["brilliant product", "terrible waste"]
X_test = vectorizer.transform(test_texts)
predictions = svm.predict(X_test)

print("SVM Predictions:")
for text, pred in zip(test_texts, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"  '{text}': {sentiment}")
```

### SVM with Different Kernels

```python
from sklearn.svm import SVC

# Linear kernel (same as LinearSVC but slower)
svm_linear = SVC(kernel='linear', random_state=42)

# RBF (Radial Basis Function) kernel - non-linear decision boundary
svm_rbf = SVC(kernel='rbf', random_state=42)

# Polynomial kernel
svm_poly = SVC(kernel='poly', degree=3, random_state=42)

# Compare
for name, model in [('Linear', svm_linear), ('RBF', svm_rbf), ('Poly', svm_poly)]:
    model.fit(X, labels)
    accuracy = model.score(X_test, predictions)
    print(f"{name:10} SVM: {accuracy:.3f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search with cross-validation
svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X, labels)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
```

## Decision Trees and Random Forests

### Decision Tree Classifier

```python
from sklearn.tree import DecisionTreeClassifier

# Train decision tree
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X, labels)

# Predictions
predictions = dt.predict(X_test)

print(f"Decision Tree accuracy: {dt.score(X_test, predictions):.3f}")
```

### Visualizing Decision Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Plot tree (limited depth for readability)
dt_simple = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_simple.fit(X, labels)

plt.figure(figsize=(20, 10))
plot_tree(
    dt_simple,
    feature_names=vectorizer.get_feature_names_out(),
    class_names=['Negative', 'Positive'],
    filled=True,
    rounded=True
)
plt.title("Decision Tree for Text Classification")
plt.show()
```

### Random Forest

Ensemble of decision trees for better performance.

```python
from sklearn.ensemble import RandomForestClassifier

# Train random forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, labels)

# Predictions with confidence
predictions = rf.predict(X_test)
probabilities = rf.predict_proba(X_test)

print("Random Forest Predictions:")
for text, pred, prob in zip(test_texts, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    print(f"  '{text}'")
    print(f"  Prediction: {sentiment} (confidence: {confidence:.3f})")
```

### Feature Importance

```python
def show_feature_importance(vectorizer, rf_model, n=15):
    """Show most important features in random forest."""
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_model.feature_importances_

    # Get top features
    top_indices = np.argsort(importances)[-n:][::-1]

    print("Top features by importance:")
    for idx in top_indices:
        if importances[idx] > 0:
            print(f"  {feature_names[idx]:15}: {importances[idx]:.4f}")

show_feature_importance(vectorizer, rf)
```

## K-Nearest Neighbors (KNN)

### Concept

Classify based on majority vote of K nearest neighbors.

```
Query point → Find K nearest neighbors → Majority class

Example (K=3):
  Query: "good product"
  Nearest neighbors:
    1. "great product" (positive)
    2. "excellent item" (positive)
    3. "nice quality" (positive)
  → Prediction: Positive (3/3 votes)
```

### Implementation

```python
from sklearn.neighbors import KNeighborsClassifier

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, labels)

# Predict
predictions = knn.predict(X_test)
distances, indices = knn.kneighbors(X_test)

print("KNN Predictions:")
for text, pred, neighbor_indices in zip(test_texts, predictions, indices):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"  Query: '{text}'")
    print(f"  Prediction: {sentiment}")
    print(f"  Nearest neighbors: {neighbor_indices}")
    print()
```

### Choosing K

```python
from sklearn.model_selection import cross_val_score

# Try different K values
k_values = [1, 3, 5, 7, 9, 11]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X, labels, cv=3)
    mean_score = cv_scores.mean()
    scores.append(mean_score)
    print(f"K={k:2}: Mean CV accuracy = {mean_score:.3f}")

# Plot
import matplotlib.pyplot as plt
plt.plot(k_values, scores, marker='o')
plt.xlabel('K')
plt.ylabel('Cross-validation Accuracy')
plt.title('KNN: Accuracy vs K')
plt.show()
```

## Model Evaluation

### Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Make predictions
y_pred = classifier.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
```

### Confusion Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Interpretation
print("\nConfusion Matrix:")
print(f"  True Negatives:  {cm[0, 0]}")
print(f"  False Positives: {cm[0, 1]}")
print(f"  False Negatives: {cm[1, 0]}")
print(f"  True Positives:  {cm[1, 1]}")
```

### Classification Report

```python
from sklearn.metrics import classification_report

# Comprehensive report
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
print(report)
```

**Output**:

```
              precision    recall  f1-score   support

    Negative       0.85      0.90      0.87       100
    Positive       0.88      0.82      0.85        95

    accuracy                           0.86       195
   macro avg       0.86      0.86      0.86       195
weighted avg       0.86      0.86      0.86       195
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Get probability scores
y_scores = classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(f"AUC Score: {roc_auc:.3f}")
```

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

## Handling Imbalanced Data

### Problem

When classes are imbalanced (e.g., 95% negative, 5% positive):

```python
# Imbalanced dataset
texts = ["negative"] * 95 + ["positive"] * 5
labels = [0] * 95 + [1] * 5

# Naive classifier that always predicts majority class achieves 95% accuracy!
# But it's useless for detecting minority class
```

### Class Weights

```python
# Automatically balance class weights
lr_balanced = LogisticRegression(class_weight='balanced')
lr_balanced.fit(X_train, y_train)

# Manual class weights
class_weights = {0: 1, 1: 10}  # Give 10x weight to class 1
lr_weighted = LogisticRegression(class_weight=class_weights)
lr_weighted.fit(X_train, y_train)
```

### Resampling

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Oversample minority class (SMOTE)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_resampled)}")

# Train on balanced data
clf = LogisticRegression()
clf.fit(X_resampled, y_resampled)
```

### Evaluation for Imbalanced Data

```python
# Don't use accuracy for imbalanced data!
# Use: F1-score, precision-recall AUC, balanced accuracy

from sklearn.metrics import balanced_accuracy_score

balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy: {balanced_acc:.3f}")

# Focus on minority class performance
f1_minority = f1_score(y_test, y_pred, pos_label=1)
print(f"F1 Score (minority class): {f1_minority:.3f}")
```

## Multi-Class and Multi-Label Classification

### Multi-Class Classification

```python
# Example: Topic classification
topics = ['sports', 'politics', 'technology', 'entertainment']

texts = [
    "football game was exciting",
    "election results announced",
    "new smartphone released",
    "movie wins oscar",
    "basketball championship",
    "parliament votes on bill"
]
labels = [0, 1, 2, 3, 0, 1]  # Corresponds to topics

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Multi-class classification (one-vs-rest by default)
clf = LogisticRegression(multi_class='ovr')
clf.fit(X, labels)

# Predict
test_texts = ["tennis match", "new laptop launched"]
X_test = vectorizer.transform(test_texts)
predictions = clf.predict(X_test)

for text, pred in zip(test_texts, predictions):
    print(f"'{text}' → {topics[pred]}")
```

### Multi-Label Classification

Documents can have multiple labels.

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Example: Movie genres
movies = [
    "action packed thriller",
    "romantic comedy film",
    "dramatic love story",
    "action comedy"
]

# Multiple labels per document
genres = [
    ['action', 'thriller'],
    ['romance', 'comedy'],
    ['drama', 'romance'],
    ['action', 'comedy']
]

# Binarize labels
mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(genres)

print("Binarized labels:")
print(y_binary)
print(f"Classes: {mlb.classes_}")

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(movies)

# Multi-label classifier
clf = MultiOutputClassifier(LogisticRegression())
clf.fit(X, y_binary)

# Predict
test_movies = ["funny action movie"]
X_test = vectorizer.transform(test_movies)
predictions_binary = clf.predict(X_test)
predictions = mlb.inverse_transform(predictions_binary)

print(f"\n'{test_movies[0]}' → {predictions[0]}")
```

### One-vs-Rest vs One-vs-One

```python
# One-vs-Rest (OvR): N classifiers for N classes
clf_ovr = LogisticRegression(multi_class='ovr')

# One-vs-One (OvO): N*(N-1)/2 classifiers
# Each classifier distinguishes between pair of classes
clf_ovo = LogisticRegression(multi_class='multinomial', solver='lbfgs')

# For 4 classes:
# OvR: 4 classifiers
# OvO: 6 classifiers

# OvR is faster, OvO can be more accurate
```

## Practical Applications

### Sentiment Analysis Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Complete pipeline
sentiment_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
    )),
    ('classifier', LogisticRegression(
        C=1.0,
        random_state=42
    ))
])

# Training data
reviews = [
    "This product is amazing! Love it!",
    "Terrible quality, waste of money",
    "Good value for the price",
    "Disappointed with the purchase",
    "Exceeded my expectations",
    "Would not recommend"
]
sentiments = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Train
sentiment_pipeline.fit(reviews, sentiments)

# Predict new reviews
new_reviews = [
    "Best purchase ever!",
    "Poor customer service",
    "Worth the money"
]

predictions = sentiment_pipeline.predict(new_reviews)
probabilities = sentiment_pipeline.predict_proba(new_reviews)

print("Sentiment Analysis Results:\n")
for review, pred, prob in zip(new_reviews, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred]
    print(f"Review: '{review}'")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.2%})")
    print()
```

### Spam Detection

```python
# Spam detection with Naive Bayes
spam_detector = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )),
    ('classifier', MultinomialNB(alpha=0.1))
])

# Training data
emails = [
    "Congratulations! You've won a million dollars!",
    "Meeting scheduled for 3pm tomorrow",
    "CLICK HERE for free iPhone!!!",
    "Project deadline reminder",
    "Claim your prize now!!!",
    "Quarterly report attached"
]
labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']

spam_detector.fit(emails, labels)

# Test
test_emails = [
    "Win money fast click now",
    "Can we reschedule our meeting?"
]

predictions = spam_detector.predict(test_emails)

for email, pred in zip(test_emails, predictions):
    print(f"'{email}' → {pred}")
```

### Intent Classification (Chatbot)

```python
# Intent classification for chatbot
intents = [
    ("what is the weather like", "weather"),
    ("tell me the temperature", "weather"),
    ("set an alarm for 7am", "alarm"),
    ("wake me up at 8", "alarm"),
    ("play some music", "music"),
    ("turn on spotify", "music"),
    ("send a message to John", "message"),
    ("text Sarah hello", "message"),
]

texts, labels = zip(*intents)

intent_classifier = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 3))),
    ('classifier', LogisticRegression())
])

intent_classifier.fit(texts, labels)

# Test queries
queries = [
    "what's the weather today",
    "play my favorite song",
    "set alarm for 6:30am"
]

for query in queries:
    intent = intent_classifier.predict([query])[0]
    print(f"Query: '{query}' → Intent: {intent}")
```

### Language Identification

```python
# Character n-gram features for language detection
language_detector = Pipeline([
    ('char_tfidf', TfidfVectorizer(
        analyzer='char',
        ngram_range=(2, 4),
        max_features=1000
    )),
    ('classifier', LogisticRegression())
])

# Training samples
texts = [
    "Hello, how are you today?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás hoy?",
    "Hallo, wie geht es dir heute?",
    "こんにちは、お元気ですか？"
]
languages = ['en', 'fr', 'es', 'de', 'ja']

language_detector.fit(texts, languages)

# Detect language
test_texts = [
    "Good morning everyone",
    "Buenos días a todos",
    "Guten Morgen allerseits"
]

for text in test_texts:
    lang = language_detector.predict([text])[0]
    print(f"'{text}' → {lang}")
```

## Best Practices

### Pipeline Best Practices

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Build flexible pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Hyperparameter search across entire pipeline
param_grid = {
    'vectorizer__max_features': [1000, 5000, 10000],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.3f}")
```

### Text Preprocessing

```python
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """Preprocess text for classification."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean(self, text):
        """Clean text."""
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize."""
        tokens = text.split()

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]

        return ' '.join(tokens)

    def preprocess(self, text):
        """Full preprocessing pipeline."""
        text = self.clean(text)
        text = self.tokenize_and_lemmatize(text)
        return text

# Use in pipeline
preprocessor = TextPreprocessor()
texts_cleaned = [preprocessor.preprocess(text) for text in texts]
```

### Model Selection

```python
from sklearn.model_selection import cross_validate

# Compare multiple models
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': LinearSVC(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Evaluate each model
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', model)
    ])

    scores = cross_validate(
        pipeline,
        texts,
        labels,
        cv=5,
        scoring=['accuracy', 'f1_macro'],
        return_train_score=True
    )

    results[name] = {
        'test_accuracy': scores['test_accuracy'].mean(),
        'test_f1': scores['test_f1_macro'].mean(),
        'train_accuracy': scores['train_accuracy'].mean()
    }

# Display results
print("\nModel Comparison:\n")
print(f"{'Model':<20} {'Test Acc':<10} {'Test F1':<10} {'Train Acc':<10}")
print("-" * 50)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['test_accuracy']:.3f}      {metrics['test_f1']:.3f}      {metrics['train_accuracy']:.3f}")
```

### Error Analysis

```python
def analyze_errors(classifier, vectorizer, X_test, y_test, texts_test):
    """Analyze classification errors."""
    predictions = classifier.predict(X_test)

    # Find misclassified examples
    errors = []
    for idx, (true_label, pred_label, text) in enumerate(zip(y_test, predictions, texts_test)):
        if true_label != pred_label:
            # Get prediction confidence
            probs = classifier.predict_proba(X_test[idx])
            confidence = probs[0][pred_label]

            errors.append({
                'text': text,
                'true': true_label,
                'predicted': pred_label,
                'confidence': confidence
            })

    # Sort by confidence (most confident errors first)
    errors.sort(key=lambda x: x['confidence'], reverse=True)

    print(f"\nFound {len(errors)} errors\n")
    print("Most confident errors:")
    for error in errors[:5]:
        print(f"Text: '{error['text']}'")
        print(f"True: {error['true']}, Predicted: {error['predicted']}")
        print(f"Confidence: {error['confidence']:.2%}")
        print()

# analyze_errors(classifier, vectorizer, X_test, y_test, texts_test)
```

## Summary

**Key Classifiers**:

1. **Naive Bayes**: Fast, works well with small data, probabilistic
2. **Logistic Regression**: Linear, interpretable, good baseline
3. **SVM**: Powerful, works well in high dimensions, kernel trick for non-linearity
4. **Random Forest**: Ensemble, robust, feature importance
5. **KNN**: Instance-based, no training, sensitive to K choice

**Feature Engineering**:

- **Bag-of-Words**: Simple word counts
- **TF-IDF**: Weight terms by importance
- **N-grams**: Capture phrases
- **Character n-grams**: Handle typos, language detection
- **Custom features**: Document length, punctuation, etc.

**Evaluation Metrics**:

- **Accuracy**: Overall correctness (not for imbalanced data)
- **Precision**: Correct positive predictions / all positive predictions
- **Recall**: Correct positive predictions / all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Classifier performance across thresholds
- **Confusion Matrix**: Detailed error breakdown

**Best Practices**:

1. Always use train-test split or cross-validation
2. Tune hyperparameters with grid/random search
3. Use pipelines for reproducibility
4. Start with simple models (Naive Bayes, Logistic Regression)
5. Consider class imbalance (class weights, resampling)
6. Analyze errors to improve model
7. Use appropriate metrics for your task
8. Preprocess text consistently

**Pipeline Template**:

```python
Pipeline([
    ('preprocessing', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])
```

**When to Use Each Classifier**:

| Classifier          | Best For                             | Avoid When                   |
| ------------------- | ------------------------------------ | ---------------------------- |
| Naive Bayes         | Small data, text, fast baseline      | Strong feature dependencies  |
| Logistic Regression | Binary/multi-class, interpretability | Non-linear decision boundary |
| SVM                 | High-dimensional data, clear margin  | Large datasets (slow)        |
| Random Forest       | Complex interactions, robustness     | Interpretability needed      |
| KNN                 | Small data, concept drift            | High dimensions, large data  |

## Next Steps

- Explore [Embeddings](../embeddings/) for dense semantic representations beyond bag-of-words
- Study [Language Models](../language_models/) to understand neural approaches to text classification
- Learn [Transformer models](../llm_concepts/) like BERT for state-of-the-art classification
- Apply [Prompt Engineering](../prompt_engineering/) for zero-shot classification with LLMs
- Study [Evaluation Metrics](../evaluation/metrics-and-benchmarks.md) for comprehensive assessment
- Explore [Fine-tuning](../application_patterns/fine-tuning-strategies.md) pre-trained models for classification tasks
