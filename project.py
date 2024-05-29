import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


ham_data = pd.read_csv('HAM10000_metadata.csv')

# Assuming your dataset has columns for features and labels
features = ham_data[['age', 'sex']]
labels = ham_data['dx']

# Convert 'sex' to numeric values
le_sex = LabelEncoder()
features.loc[:, 'sex'] = le_sex.fit_transform(features['sex'])  # Use .loc to avoid SettingWithCopyWarning

plt.figure(figsize=(10, 6))
sns.countplot(x='sex', data=ham_data)
plt.title('Distribution of Lesions by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.show()
plt.close()

# Visualize based on Age
plt.figure(figsize=(12, 6))
sns.histplot(x='age', data=ham_data, bins=20, kde=True, color='skyblue')
plt.title('Distribution of Lesions by Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
plt.close()

# Visualize based on Localization
plt.figure(figsize=(14, 8))
sns.countplot(x='localization', data=ham_data)
plt.title('Distribution of Lesions by Localization')
plt.xlabel('Localization')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.show()
plt.close()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(ham_data[['age']], ham_data['sex'], test_size=0.2, random_state=42)

# Impute missing values (replace NaN with the mean)
imputer = SimpleImputer(strategy='mean')
X_train['age'] = imputer.fit_transform(X_train[['age']])
X_test['age'] = imputer.transform(X_test[['age']])

# Reshape the data for CNN compatibility
X_train_reshaped = np.asarray(X_train['age']).reshape(-1, 1)
X_test_reshaped = np.asarray(X_test['age']).reshape(-1, 1)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Label encode the target variable 'sex'
le_sex = LabelEncoder()
y_train_encoded = le_sex.fit_transform(y_train)
y_test_encoded = le_sex.transform(y_test)

# One-hot encode labels
y_train_onehot = keras.utils.to_categorical(y_train_encoded, num_classes=len(le_sex.classes_))
y_test_onehot = keras.utils.to_categorical(y_test_encoded, num_classes=len(le_sex.classes_))

# Build a simple CNN model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(1,)),
    keras.layers.Dense(len(le_sex.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Lists to store accuracy values for each epoch
train_accuracy = []
test_accuracy = []

# Train the model and store accuracy values
for epoch in range(10):
    history = model.fit(X_train_scaled, y_train_onehot, epochs=1, batch_size=32, validation_split=0.1)
    
    # Extract accuracy values from the training history
    train_accuracy.append(history.history['accuracy'][0])
    test_accuracy.append(history.history['val_accuracy'][0])

# Plot the accuracy values
plt.plot(range(1, 11), train_accuracy, label='Train Accuracy')
plt.plot(range(1, 11), test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()

# Evaluate the model on the test set
y_pred_onehot = model.predict(X_test_scaled)
y_pred_encoded = np.argmax(y_pred_onehot, axis=1)
y_pred = le_sex.inverse_transform(y_pred_encoded)

# Evaluate the performance of the classifier
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))


