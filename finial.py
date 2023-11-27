import numpy as np
import pandas as pd
import os
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import re


# The directory where our MRI .nii files are located!!!!
mri_image_dir = '../ADNI_RESIZED2' 

# Use os.listdir -->to get all file names in this directory
#Don't get the name one by one, which is what I did before
mri_image_paths = [os.path.join(mri_image_dir, fname) for fname in os.listdir(mri_image_dir) if fname.endswith('.nii')]


#print("Current working directory:", os.getcwd())
#print("Contents of the MRI directory:", os.listdir(mri_image_dir))
#print("Number of MRI files:", len(mri_image_paths))
#print("First 10 paths:", mri_image_paths[:10])

# 3D CNN architecture
def create_3d_cnn_model(width, height, depth, num_classes):
    inputShape = (width, height, depth, 1) 
    inputs = tf.keras.Input(inputShape)
    
    x = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

model = create_3d_cnn_model(64, 64, 64, num_classes=2)  # num_classes=2 --> binary classification

# Load and preprocess an MRI image
def preprocess_and_extract_features_batch(file_paths, model):
    batch_data = []
    for file_path in file_paths:
        # Load the MRI image (.nii file) using nibabel
        image_data = nib.load(file_path)
        data = image_data.get_fdata()
        # Normalize the data and preprocess
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = np.resize(data, (64, 64, 64))  # Resize
        batch_data.append(data)
    
    batch_data = np.array(batch_data)
    batch_data = np.expand_dims(batch_data, axis=-1) 

    # 3D CNN to extract features that we need
    features = model.predict(batch_data)
    return features.reshape((features.shape[0], -1))  # Reshape

# process images one by one
#features = np.array([preprocess_and_extract_features(path, model) for path in mri_image_paths])

#-------------------------------------------------------------------------------------------------------------
# process images in batches
batch_size = 10
features_list = []

for i in range(0, len(mri_image_paths), batch_size):
    batch_paths = mri_image_paths[i:i + batch_size]
    features = preprocess_and_extract_features_batch(batch_paths, model)
    features_list.append(features)

# Concatenate all batches to get the full features array
features = np.concatenate(features_list, axis=0)
#-------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------
# Save the features to a CSV file
np.savetxt('../extracted_features.csv', features, delimiter=',')  
#-------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------
# ... [Your previous code for feature extraction]

# Read the labels CSV file
labels_df = pd.read_csv('../Labels.csv')
print(labels_df.head())

# Create a dictionary mapping from Image Data ID to label
labels_dict = dict(zip(labels_df['Image Data ID'], labels_df['Group']))
print(list(labels_dict.items())[:5])

# Define a function to extract the Image Data ID from the file name
def extract_image_data_id(filename):
    match = re.match(r'.*_(I\d+)\.nii', filename)
    if match:
        return match.group(1)
    return None  # Return None if the pattern does not match

# Extract the Image Data ID from each filename
mri_image_ids = [extract_image_data_id(os.path.basename(path)) for path in mri_image_paths]
print(list(mri_image_ids)[:5])

# Debug: Print the first few Image Data IDs and labels
mri_image_paths = [os.path.join(mri_image_dir, fname) for fname in os.listdir(mri_image_dir) if fname.endswith('.nii')]
mri_image_labels = [labels_dict.get(extract_image_data_id(os.path.basename(path))) for path in mri_image_paths]
for path, label in zip(mri_image_paths[:5], mri_image_labels[:5]):
    print(f'File: {path} - Label: {label}')

# Ensure that we only use images for which we have labels
# Create a list for features and labels where labels are available
filtered_features = []
filtered_labels = []
for image_id, feature in zip(mri_image_ids, features):
    label = labels_dict.get(image_id)
    if label is not None:
        filtered_features.append(feature)
        filtered_labels.append(label)

# Convert to numpy arrays
filtered_features = np.array(filtered_features)
filtered_labels = np.array(filtered_labels)

# Debug
print("First 10 feature-label pairs:")
for i in range(min(10, len(filtered_features))):  # Ensures we don't go out of bounds
    print(f"Feature {i}:", filtered_features[i])
    print(f"Label {i}:", filtered_labels[i])


# Now, filtered_features and filtered_labels have the same length
# and can be used for training and testing

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(filtered_features, filtered_labels, test_size=0.2, random_state=42)
#-------------------------------------------------------------------------------------------------------------



# Train a Random Forest Classifier on the features extracted by the 3D CNN
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Save the predictions to a CSV file
#np.savetxt('predictions.csv', y_pred, delimiter=',')
np.savetxt('predictions.csv', y_pred, fmt='%s', delimiter=',')


# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the classification report to a text file
with open('classification_report.txt', 'w') as f:
    f.write(report)
# Save the model's accuracy to a text file
with open('model_accuracy.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")

# Output the accuracy and classification report
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
