from pathlib import Path
import numpy as np

# Dirs, Flags, Constants
dir = Path(__file__).resolve().parent.parent / "data"

saveFlag = False

stepSize = 0.01
limit = 10000

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost_function(weights, X, y):
    sigma = sigmoid(np.dot(X, weights))
    cost = -np.sum(y * np.log(sigma) + (1 - y) * np.log(1 - sigma))
    
    return cost

def compute_gradient(weights, X, y):
    z = np.dot(X, weights)
    sigma = sigmoid(z)
    gradient = np.dot(X.T, (sigma - y)) / len(y)

    return gradient

def gradient_descent(X, y):
    # Create weights
    weights = np.zeros(X.shape[1])
    loss_history = []

    for i in range(limit):
        gradient = compute_gradient(weights, X, y)
        weights -= (stepSize * gradient)

        loss_history.append(cost_function(weights, X, y))

        if i % 1000 == 0:
            print(f"Iteration {i}: Current Loss: {loss_history[-1]}")

    return weights

def evaluate(weights, X, y):
    z = np.dot(X, weights)
    prob = sigmoid(z)

    predictions = (prob >= 0.5).astype(int)
    accuracy = np.mean(predictions == y)

    return accuracy

def ec_evaluate(weights, X):
    z = np.dot(X, weights)
    prob = sigmoid(z)

    return prob
    
def main():
    # Load Data
    train_x = np.load(dir / "train4_2_ec.npz")["x"]
    train_y = np.load(dir / "train4_2_ec.npz")["y"]

    test_x = np.load(dir / "test4_2_ec.npz")["x"]
    test_y = np.load(dir / "test4_2_ec.npz")["y"]

    # Train model
    weights = gradient_descent(train_x, train_y)

    # Evaluate on Train/Test Set 
    train_acc = evaluate(weights, train_x, train_y)
    test_acc = evaluate(weights, test_x, test_y)

    print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    # Load mail.txt, preprocess
    mail = np.load(dir / "mail_ec.npz")["x"]

    # Evaluate
    val_mail = ec_evaluate(weights, mail)
    
    print(f"Sigmoid Probability: {val_mail}, Spam? {round(val_mail[0])}")
    
main()