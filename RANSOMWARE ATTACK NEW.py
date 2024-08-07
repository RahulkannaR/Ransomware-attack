import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from keras.layers import Input
from keras.models import Model
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("D:\data_file.csv")

# Select only the specified columns
df = df[["Machine", "DebugSize", "NumberOfSections", "SizeOfStackReserve", "MajorOSVersion", "BitcoinAddresses", "Benign", "DebugRVA", "MajorImageVersion", "ExportRVA", "ExportSize", "IatVRA", "MajorLinkerVersion", "MinorLinkerVersion", "DllCharacteristics", "ResourceSize"]]

# Prepare the data
X = df.drop(columns=['Benign'])
y = df['Benign']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CNN-LSTM Model
cnn_lstm_model_input = Input(shape=(X_train_scaled.shape[1], 1))
cnn_lstm_model_layers = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_lstm_model_input)
cnn_lstm_model_layers = MaxPooling1D(pool_size=2)(cnn_lstm_model_layers)
cnn_lstm_model_layers = LSTM(units=50)(cnn_lstm_model_layers)
cnn_lstm_model_output = Dense(1, activation='sigmoid')(cnn_lstm_model_layers)
cnn_lstm_model = Model(inputs=cnn_lstm_model_input, outputs=cnn_lstm_model_output)

cnn_lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the CNN-LSTM model
cnn_lstm_model.fit(X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1), y_train, epochs=10, batch_size=32, verbose=1)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# GBM Model
gbm_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm_model.fit(X_train_scaled, y_train)

# Evaluate the models
# CNN-LSTM model evaluation
cnn_lstm_predictions = (cnn_lstm_model.predict(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)) > 0.5).astype("int32")

# Random Forest model evaluation
rf_predictions = rf_model.predict(X_test_scaled)

# GBM model evaluation
gbm_predictions = gbm_model.predict(X_test_scaled)

# Combined predictions
combined_predictions = (cnn_lstm_predictions.flatten() + rf_predictions + gbm_predictions) / 3
combined_predictions[combined_predictions <= 0.5] = 0
combined_predictions[combined_predictions > 0.5] = 1

# Calculate metrics
# Individual model metrics
cnn_lstm_accuracy = accuracy_score(y_test, cnn_lstm_predictions)
cnn_lstm_classification_report = classification_report(y_test, cnn_lstm_predictions)
cnn_lstm_confusion_matrix = confusion_matrix(y_test, cnn_lstm_predictions)
cnn_lstm_roc_auc = roc_auc_score(y_test, cnn_lstm_predictions)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)
rf_roc_auc = roc_auc_score(y_test, rf_predictions)

gbm_accuracy = accuracy_score(y_test, gbm_predictions)
gbm_classification_report = classification_report(y_test, gbm_predictions)
gbm_confusion_matrix = confusion_matrix(y_test, gbm_predictions)
gbm_roc_auc = roc_auc_score(y_test, gbm_predictions)

# Combined model metrics
combined_accuracy = accuracy_score(y_test, combined_predictions)
combined_classification_report = classification_report(y_test, combined_predictions)
combined_confusion_matrix = confusion_matrix(y_test, combined_predictions)
combined_roc_auc = roc_auc_score(y_test, combined_predictions)

# Print results
print("CNN-LSTM Model Metrics:")
print("Accuracy:", cnn_lstm_accuracy)
print("Classification Report:")
print(cnn_lstm_classification_report)
print("Confusion Matrix:")
print(cnn_lstm_confusion_matrix)
print("ROC AUC Score:", cnn_lstm_roc_auc)
print()

print("Random Forest Model Metrics:")
print("Accuracy:", rf_accuracy)
print("Classification Report:")
print(rf_classification_report)
print("Confusion Matrix:")
print(rf_confusion_matrix)
print("ROC AUC Score:", rf_roc_auc)
print()

print("GBM Model Metrics:")
print("Accuracy:", gbm_accuracy)
print("Classification Report:")
print(gbm_classification_report)
print("Confusion Matrix:")
print(gbm_confusion_matrix)
print("ROC AUC Score:", gbm_roc_auc)
print()

print("Combined Model Metrics:")
print("Accuracy:", combined_accuracy)
print("Classification Report:")
print(combined_classification_report)
print("Confusion Matrix:")
print(combined_confusion_matrix)
print("ROC AUC Score:", combined_roc_auc)

# Visualizations
# Confusion Matrix Heatmaps
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.heatmap(cnn_lstm_confusion_matrix, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
axes[0, 0].set_title('CNN-LSTM Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

sns.heatmap(rf_confusion_matrix, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues')
axes[0, 1].set_title('Random Forest Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

sns.heatmap(gbm_confusion_matrix, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
axes[1, 0].set_title('GBM Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')

sns.heatmap(combined_confusion_matrix, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
axes[1, 1].set_title('Combined Model Confusion Matrix')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ROC Curves
plt.figure(figsize=(10, 6))
fpr, tpr, _ = roc_curve(y_test, cnn_lstm_predictions)
plt.plot(fpr, tpr, label='CNN-LSTM (area = %0.2f)' % cnn_lstm_roc_auc)
fpr, tpr, _ = roc_curve(y_test, rf_predictions)
plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
fpr, tpr, _ = roc_curve(y_test, gbm_predictions)
plt.plot(fpr, tpr, label='GBM (area = %0.2f)' % gbm_roc_auc)
fpr, tpr, _ = roc_curve(y_test, combined_predictions)
plt.plot(fpr, tpr, label='Combined Model (area = %0.2f)' % combined_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

# Feature Importance
rf_importances = rf_model.feature_importances_
gbm_importances = gbm_model.feature_importances_
features = X.columns

plt.tight_layout()
plt.show()

# Sample data visualization
attack_types = ['Phishing Emails', 'Exploit Kits', 'Remote Desktop Protocol', 'Drive-by Downloads']
attack_counts = [120, 90, 60, 30]  # Example attack counts

# Create bar plot
plt.figure(figsize=(10, 6))
plt.barh(attack_types, attack_counts, color='skyblue')
plt.xlabel('Number of Attacks')
plt.ylabel('Attack Types')
plt.title('Ransomware Attack Distribution')
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.show()
