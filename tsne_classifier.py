import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE, LocallyLinearEmbedding
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
import seaborn as sns
import umap

print("Step 1: Loading the MNIST dataset...")
# Load the MNIST dataset using scikit-learn
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist.data, mnist.target.astype(int)
print(f"Loaded MNIST dataset with {x.shape[0]} samples.")

# Split the dataset into training and test sets
print("Step 2: Splitting dataset into training and test sets...")
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]
print(f"Training set size: {x_train.shape[0]} samples, Test set size: {x_test.shape[0]} samples.")

# Convert to NumPy arrays if needed
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()

# Available reduction methods
print("Step 3: Setting up dimensionality reduction methods...")
methods = {
    "tsne": TSNE(n_components=2, random_state=42),
    "umap": umap.UMAP(n_components=2, random_state=42) if 'umap' in globals() else None,
    "pca": PCA(n_components=2),
    "lle": LocallyLinearEmbedding(n_components=2)
}

# Select methods to use (you can customize this list)
selected_methods = ["tsne", "umap", "pca", "lle"]
selected_methods = ["umap", "pca", "lle"]

# Dictionary to store results for comparison
results = {}

print(f"Step 4: Starting comparison of methods: {', '.join(selected_methods)}")

# Loop over each selected method
for method_name in selected_methods:
    print(f"\nUsing {method_name.upper()} for dimensionality reduction...")

    if method_name not in methods or methods[method_name] is None:
        print(f"{method_name.upper()} is not available or not installed. Skipping.")
        continue

    # Initialize the model
    model = methods[method_name]
    print(f"Initialized {method_name.upper()} model.")

    # Initialize an empty list to store predicted labels
    predicted_labels = []

    # Iterate over each test sample and perform dimensionality reduction with it included in the training set
    for i, test_sample in enumerate(x_test):
        print(f"Processing test sample {i + 1}/{len(x_test)}...", end='\r')

        # Reshape the test sample to match the training data format (1, -1)
        test_sample = test_sample.reshape(1, -1)

        # Combine the test sample with the training set
        combined_data = np.vstack([x_train, test_sample])
        print(f"  Combined test sample {i + 1} with training set.")

        # Apply the dimensionality reduction method
        print(f"  Applying {method_name.upper()} to the combined data...")
        combined_reduced = model.fit_transform(combined_data)
        print(f"  {method_name.upper()} reduction completed.")

        # Use K-means to cluster the reduced data
        print(f"  Applying K-means clustering on the reduced data...")
        kmeans = KMeans(n_clusters=10, random_state=42)
        combined_labels = kmeans.fit_predict(combined_reduced)
        print(f"  K-means clustering completed.")

        # Extract the cluster assignment for the test sample (last index)
        test_sample_cluster = combined_labels[-1]  # Last point is the test sample
        print(f"  Test sample {i + 1} assigned to cluster {test_sample_cluster}.")

        # Get training labels corresponding to the same cluster as the test sample
        train_labels_in_cluster = y_train[combined_labels[:-1] == test_sample_cluster]
        print(f"  Found {len(train_labels_in_cluster)} training samples in the same cluster.")

        # Assign the test sample the most common label within its cluster
        if len(train_labels_in_cluster) > 0:
            predicted_label = Counter(train_labels_in_cluster).most_common(1)[0][0]
        else:
            # If the cluster is empty, choose a random label (this is a fallback)
            predicted_label = np.random.choice(y_train)
        print(f"  Assigned label {predicted_label} to test sample {i + 1}.")

        # Store the predicted label
        predicted_labels.append(predicted_label)

    # Calculate the accuracy of this method
    accuracy = accuracy_score(y_test, predicted_labels)
    results[method_name] = {
        "predicted_labels": predicted_labels,
        "accuracy": accuracy
    }
    print(f"\nAccuracy of {method_name.upper()} + K-means labeling approach: {accuracy:.4f}")

# Compare results
print("\nStep 5: Comparison of Dimensionality Reduction Methods:")
for method_name, result in results.items():
    print(f"{method_name.upper()}: Accuracy = {result['accuracy']:.4f}")

# Plot confusion matrices for each method
print("\nStep 6: Plotting confusion matrices...")
for method_name, result in results.items():
    conf_matrix = confusion_matrix(y_test, result["predicted_labels"])
    print(f"  Plotting confusion matrix for {method_name.upper()}...")

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title(f'Confusion Matrix - {method_name.upper()}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    print(f"  Confusion matrix for {method_name.upper()} displayed.")
