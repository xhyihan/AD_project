import numpy as np
import pandas as pd
import os
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# 3D CNN 
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


def preprocess_mri(file_path):
    image_data = nib.load(file_path)
    data = image_data.get_fdata()
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = np.resize(data, (64, 64, 64))  # Resize
    return data


mri_image_dir = '../ADNI_RESIZED2'
mri_image_paths = [os.path.join(mri_image_dir, fname) for fname in os.listdir(mri_image_dir) if fname.endswith('.nii')]


labels_df = pd.read_csv('../Labels.csv')
labels_dict = dict(zip(labels_df['Image Data ID'], labels_df['Group']))


mri_image_labels = [labels_dict.get(re.match(r'.*_(I\d+)\.nii', os.path.basename(path)).group(1)) for path in mri_image_paths]


mri_data = [(preprocess_mri(path), label) for path, label in zip(mri_image_paths, mri_image_labels) if label is not None]
mri_images, mri_labels = zip(*mri_data)

label_to_index = {'AD': 0, 'CN': 1, 'MCI': 2}
mri_labels = [label_to_index[label] for label in mri_labels]


mri_images = np.array(mri_images)
mri_labels = np.array(mri_labels)


mri_images = np.expand_dims(mri_images, axis=-1)


X_train, X_test, y_train, y_test = train_test_split(mri_images, mri_labels, test_size=0.2, random_state=42)


model = create_3d_cnn_model(64, 64, 64, num_classes=3)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)


accuracy = accuracy_score(y_test, y_pred_classes)
report = classification_report(y_test, y_pred_classes)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
