import ecdsa
import base58
import hashlib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import os

# Function to generate a random private key
def generate_private_key():
    while True:
        private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1).to_string().hex()
        # Validate the generated private key
        if len(private_key) == 64:  # Check if the length is 64 characters
            return private_key

# Function to derive the public key from the private key
def private_key_to_public_key(private_key):
    try:
        sk = ecdsa.SigningKey.from_string(bytes.fromhex(private_key), curve=ecdsa.SECP256k1)
        vk = sk.get_verifying_key()
        return vk.to_string().hex()
    except ecdsa.keys.BadDigestError:
        print(f"Invalid private key: {private_key}")
        return None

# Function to derive the Bitcoin address from the public key
def public_key_to_address(public_key):
    try:
        public_key_bytes = bytes.fromhex(public_key)
    except ValueError:
        return None

    sha256_hash = hashlib.sha256(public_key_bytes).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    return base58.b58encode_check(b'\x00' + ripemd160_hash).decode('utf-8')

# Function to create a new wallet and private key
def create_wallet():
    while True:
        private_key = generate_private_key()
        public_key = private_key_to_public_key(private_key)
        if public_key:
            address = public_key_to_address(public_key)
            if address:
                return private_key, public_key, address

# Function to convert keys to feature vectors
def keys_to_features(keys):
    return np.array([[int(char, 16) for char in key] for key in keys], dtype=np.float32)

# Function to train and save the model
def train_and_save_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Function to verify ownership of a wallet address
def verify_ownership(model, scaler, private_keys, provided_address):
    features = keys_to_features(private_keys)

    # Fit the scaler with the training data
    scaler.fit(features)

    features = scaler.transform(features)
    predictions = model.predict(features)

    # Convert numpy.int32 values to hexadecimal strings
    predictions_hex = [hex(prediction)[2:] for prediction in predictions]

    # Find the corresponding public key for the provided address
    provided_public_key = [public_key for public_key, address in zip(predictions_hex, private_keys) if public_key_to_address(public_key) == provided_address]

    if provided_public_key:
        return True, provided_public_key[0]
    else:
        return False, None
        
# Function to store evaluation metrics in a file
def store_evaluation_metrics(metrics, file_path):
    with open(file_path, 'a') as f:
        f.write("Iteration Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")
        
def fine_tune_hyperparameters(X_train, y_train):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Create a RandomForestClassifier
    base_model = RandomForestClassifier(random_state=42)

    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model from the search
    best_model = grid_search.best_estimator_

    print("Best Hyperparameters:")
    print(grid_search.best_params_)

    return best_model

# Specify the file path and model file
file_path = 'dataset.pkl'
model_file_path = 'ownership_verification_model.pkl'

# Specify the number of times to add new training data in each iteration
iterations = 1

# Specify the file path for storing metrics
metrics_file_path = 'evaluation_metrics.txt'

# Initialize dataset
dataset = {'private_keys': [], 'public_keys': [], 'addresses': [], 'labels': []}

try:
    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
        print("Loaded existing dataset.")
except (FileNotFoundError, EOFError):
    print("Dataset not found. Creating a new dataset.")

# Load or create the model
try:
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
        print("Model loaded.")
except (FileNotFoundError, EOFError):
    print("Model not found. Creating a new model.")
    model = RandomForestClassifier(n_estimators=400, max_depth=30, min_samples_split=2, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

print(f"Generating new wallets")

for _ in range(iterations):
    # Generate and add new training data
    for _ in range(1000):
        private_key, public_key, address = create_wallet()
        label = int(private_key[-1], 16) % 2
        dataset['private_keys'].append(private_key)
        dataset['public_keys'].append(public_key)
        dataset['addresses'].append(address)
        dataset['labels'].append(label)

    # Train the model and save it
    X_train, X_test, y_train, y_test = train_test_split(keys_to_features(dataset['private_keys']), dataset['labels'], test_size=0.2, random_state=42)

    # Fine-tune Hyperparameters
    model = fine_tune_hyperparameters(X_train, y_train)

    # Fit the scaler with the training data
    scaler.fit(X_train)

    # Transform the test data
    X_test_transformed = scaler.transform(X_test)

    # Calculate accuracy on test data
    print(f"Trying to predict")
    test_predictions = model.predict(X_test_transformed)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Accuracy on test data: {test_accuracy * 100:.2f}%")

    # Calculate precision, recall, and F1 score
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1_score = f1_score(y_test, test_predictions)

    print(f"Precision on test data: {test_precision * 100:.2f}%")
    print(f"Recall on test data: {test_recall * 100:.2f}%")
    print(f"F1 Score on test data: {test_f1_score * 100:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, test_predictions)
    print("Confusion Matrix:")
    print(cm)

    # Store Evaluation Metrics
    iteration_metrics = {
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'F1 Score': test_f1_score
    }
    store_evaluation_metrics(iteration_metrics, metrics_file_path)

    # Save the updated dataset
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)
        print("Dataset saved.")

    # Save the updated model
    with open(model_file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
        print("Model saved.")

    # Verify ownership for a provided wallet address
    provided_wallet_address = "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
    ownership_status, corresponding_public_key = verify_ownership(model, scaler, dataset['private_keys'], provided_wallet_address)

    if ownership_status:
        print(f"Iteration {_ + 1}: The provided wallet address is owned by the corresponding public key: {corresponding_public_key}")
