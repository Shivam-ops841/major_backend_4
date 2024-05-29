import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the dataset into a DataFrame named 'df'
df = pd.read_csv('HAM10000_metadata.csv')

# Display basic information about the dataset
print("Dataset Overview:")
print(df.info())

# Visualize the distribution of the target label ('dx_type')
plt.figure(figsize=(8, 6))
sns.countplot(x='dx_type', data=df)
plt.title('Distribution of Diagnosis Type')
plt.show()

# Identify features and target label
features_cols = df.columns.tolist()  # Use all columns as features

# Choose the target label ('dx_type')
target_label = 'dx_type'
if target_label in features_cols:
    features_cols.remove(target_label)  # Remove the target label column

# Create a new DataFrame for features
features = df[features_cols].copy()

# Encode categorical variables to numeric if needed
for col in features.columns:
    if features[col].dtype == 'object':
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_imputed, df[target_label], test_size=0.2, random_state=42)

# Create a Naive Bayes classifier (Gaussian Naive Bayes)
naive_bayes_classifier = GaussianNB()

# Train the classifier
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = naive_bayes_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f'\nAccuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)


precision_optimal = precision_score(y_test, y_pred, average='weighted')
recall_optimal = recall_score(y_test, y_pred, average='weighted')
f1_optimal = f1_score(y_test, y_pred, average='weighted')

print(f'\nWeighted Precision for Optimal k: {precision_optimal:.2f}')
print(f'Weighted Recall for Optimal k: {recall_optimal:.2f}')
print(f'Weighted F1 Score for Optimal k: {f1_optimal:.2f}')