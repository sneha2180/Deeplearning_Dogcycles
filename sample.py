import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model

# Define the estrous stages
estrous_stages = ['Anestrus', 'Estrous', 'Diestrus', 'Proestrus']

# Path to the folder containing your local images for each estrous stage
image_folders = ['Database_Deep learning/Anestrus', 'Database_Deep learning/Estrus', 'Database_Deep learning/Diestrus', 'Database_Deep learning/Prestrus']

# Initialize lists to store extracted features for each estrous stage
features_lists = []

# Load the pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed7').output)

# Process local image files for each estrous stage
for folder in image_folders:
    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    features_list = []
    for image_file in image_files:
        img_path = os.path.join(folder, image_file)
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features_list.append(features)
    features_lists.append(features_list)

# Create a heatmap from the features
features_arrays = [np.vstack(features_list) for features_list in features_lists]

# Combine the features arrays and reshape for PCA
features_array = np.vstack(features_arrays)
features_array = features_array.reshape(-1, features_array.shape[-1])

# Apply PCA to reduce dimensionality
pca = PCA(n_components=4)
features_pca = pca.fit_transform(features_array)

# Display the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(features_pca, annot=False, cmap="YlGnBu", xticklabels=estrous_stages)
plt.title("Heatmap of Image Features (Reduced Dimensionality)")
plt.show()