"""fonction de la descente de gradient et Newton """
#Fonction Titanic
# Fonction de coût et gradient
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_cost(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1/m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))

def logistic_gradient(theta, X, y):
    m = len(y)
    h = sigmoid(X @ theta)
    return (1/m) * X.T @ (h - y)

def logistic_hessian(theta, X):
    m = X.shape[0]
    h = sigmoid(X @ theta)
    S = np.diag(h * (1 - h))
    return (1/m) * (X.T @ S @ X)

# Gradient Descent logistique
def logistic_gradient_descent(X, y, alpha=0.01, max_iter=100, tol=1e-6):
    theta = np.zeros(X.shape[1])
    costs = []
    for i in range(max_iter):
        grad = logistic_gradient(theta, X, y)
        theta -= alpha * grad
        cost = logistic_cost(theta, X, y)
        costs.append(cost)
        if i > 0 and abs(costs[-2] - costs[-1]) < tol:
            break
    return theta, costs

# Newton's Method logistique
def logistic_newton_method(X, y, max_iter=100, tol=1e-6):
    theta = np.zeros(X.shape[1])
    costs = []
    for i in range(max_iter):
        grad = logistic_gradient(theta, X, y)
        hess = logistic_hessian(theta, X)

      # Mettre à jour theta via la méthode de Newton
        try:
            theta -= np.linalg.inv(hess).dot(grad)
        except np.linalg.LinAlgError:
            print("Erreur lors de l'inversion de la Hessienne.")
            break    
        cost = logistic_cost(theta, X, y)
        costs.append(cost)
        if i > 0 and abs(costs[-2] - costs[-1]) < tol:
            break
    return theta, costs

#3. Fonctions pour load digit

#2. Fonctions de Boston
# Descente de gradient
def gradient_descent(X, y, lr=0.1, iterations=1000, tol=1e-6):
    m = len(y)
    theta = np.zeros(X.shape[1])
    history = []
    
    for i in range(iterations):
        gradients = (2/m) * X.T @ (X @ theta - y)
        theta -= lr * gradients
        history.append(np.mean((X @ theta - y)**2))  # MSE
        if i > 0 and abs(history[-2] - history[-1]) < tol:
            break
    return theta, history

# Méthode de Newton
def newton_method(X, y, iterations=100, tol=1e-6):
    m = len(y)
    theta = np.zeros(X.shape[1])
    history = []
    
    for i in range(iterations):
        gradients = (2/m) * X.T @ (X @ theta - y)
        hessian = (2/m) * X.T @ X
        try:
            theta -= np.linalg.inv(hessian) @ gradients
        except:
            print("Erreur lors de l'inversion de la Hessienne.")
            break 

        history.append(np.mean((X @ theta - y)**2))  # MSE

        if i > 0 and abs(history[-2] - history[-1]) < tol:
            break
    return theta, history
