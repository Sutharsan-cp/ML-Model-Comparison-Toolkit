import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Linear Regression Classes
# ---------------------------

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class RidgeLinearRegression(LinearRegression):
    def __init__(self, lr=0.01, epochs=1000, lamda=0.1):
        super().__init__(lr, epochs)
        self.lamda = lamda

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/m) * np.dot(X.T, (y_pred - y)) + (self.lamda/m) * self.weights
            db = (1/m) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self


class LassoLinearRegression(LinearRegression):
    def __init__(self, lr=0.01, epochs=1000, lamda=0.1):
        super().__init__(lr, epochs)
        self.lamda = lamda

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/m) * np.dot(X.T, (y_pred - y)) + (self.lamda/m) * np.sign(self.weights)
            db = (1/m) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self


class ElasticNetLinearRegression(LinearRegression):
    def __init__(self, lr=0.01, epochs=1000, lamda=0.1, alpha=0.5):
        super().__init__(lr, epochs)
        self.lamda = lamda
        self.alpha = alpha

    def train(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/m) * np.dot(X.T, (y_pred - y)) \
                 + self.alpha * (self.lamda/m) * np.sign(self.weights) \
                 + (1-self.alpha) * (self.lamda/m) * self.weights
            db = (1/m) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self


# ---------------------------
# Helper Functions
# ---------------------------

def load_datasets(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]


def prepare_data(file_path, target_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)
    return X, y


def visualize_predictions(y_raw, y_pred_raw, y_proc, y_pred_proc, title_raw='Raw Dataset', title_proc='Processed Dataset'):
    """
    Left subplot: raw dataset
    Right subplot: processed dataset
    """
    plt.figure(figsize=(12, 5))

    # Left: Raw
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_raw)), y_raw, color='blue', label='True')
    plt.scatter(range(len(y_pred_raw)), y_pred_raw, color='red', label='Predicted', alpha=0.6)
    plt.title(title_raw)
    plt.legend()

    # Right: Processed
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y_proc)), y_proc, color='blue', label='True')
    plt.scatter(range(len(y_pred_proc)), y_pred_proc, color='red', label='Predicted', alpha=0.6)
    plt.title(title_proc)
    plt.legend()

    plt.show()


# ---------------------------
# Menu-driven Execution
# ---------------------------

def main():
    # Dataset directories (fixed paths relative to script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.join(script_dir, '../../../data/processedDatasets/regression')
    raw_dir       = os.path.join(script_dir, '../../../data/rawDatasets/regression')

    # Step 1: Choose dataset
    print("Available datasets (processed):")
    processed_files = load_datasets(processed_dir)
    for i, f in enumerate(processed_files):
        print(f"{i+1}. {f}")
    choice = int(input("Select dataset by number: ")) - 1
    dataset_name = processed_files[choice]

    # Step 2: Choose target column
    target_column = input("Enter target column name: ")

    # Step 3: Load processed dataset
    X_proc, y_proc = prepare_data(os.path.join(processed_dir, dataset_name), target_column)

    # Step 4: Load raw dataset if exists
    raw_path = os.path.join(raw_dir, dataset_name)
    if os.path.exists(raw_path):
        X_raw, y_raw = prepare_data(raw_path, target_column)
        raw_available = True
    else:
        print("Raw dataset not found, skipping raw comparison.")
        # If raw dataset missing, just use processed data for left plot
        X_raw, y_raw = X_proc, y_proc
        raw_available = False

    # Step 5: Choose Linear Regression type
    print("\nSelect model:")
    models = {
        1: ('Linear Regression', LinearRegression()),
        2: ('Ridge Regression', RidgeLinearRegression(lamda=0.1)),
        3: ('Lasso Regression', LassoLinearRegression(lamda=0.1)),
        4: ('Elastic Net Regression', ElasticNetLinearRegression(lamda=0.1, alpha=0.5))
    }
    for k, v in models.items():
        print(f"{k}. {v[0]}")
    model_choice = int(input("Enter choice: "))
    model_name, model_obj = models[model_choice]

    # Step 6: Train on processed dataset
    model_obj.train(X_proc, y_proc)
    y_pred_proc = model_obj.predict(X_proc)

    # Step 7: Train on raw dataset if available
    model_obj.train(X_raw, y_raw)
    y_pred_raw = model_obj.predict(X_raw)

    # Step 8: Visualize (left: raw, right: processed)
    visualize_predictions(y_raw, y_pred_raw, y_proc, y_pred_proc,
                          title_raw='Raw Dataset', title_proc='Processed Dataset')


if __name__ == "__main__":
    main()
