import numpy as np
import matplotlib.pyplot as plt

# Model import
from sklearn.svm import SVC


class SVMClassifier:
    def __init__(self, kernel='poly',C = 100, coef0 = 10, degree= 2, gamma= 'scale', scaler=1000):
        """
        Initializes the SVMClassifier with specified kernel, regularization, and scaling parameters.

        Parameters:
        - kernel (str): The kernel type to be used in the algorithm. 
                        Options include 'linear', 'poly', 'rbf', 'sigmoid', etc. (default: 'poly').
        - C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. 
                     Must be a positive float (default: 100).
        - coef0 (float): Independent term in kernel function. Only significant for 'poly' and 'sigmoid' kernels (default: 10).
        - degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels (default: 2).
        - gamma (str or float): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. 
                                Can be 'scale', 'auto', or a positive float (default: 'scale').
        - scaler (float): A scaling factor to avoid floating point errors (default: 1000).

        Attributes:
        - model (sklearn.svm.SVC): The Support Vector Classifier instance initialized with the given parameters.
        """
        # Initialize the SVM with the given kernel, C, and degree
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

        self.svm_clf = SVC(kernel=self.kernel,C = self.C, coef0 = self.coef0, degree= self.degree, gamma=  self.gamma)

        self.scaler = scaler if not scaler is None else 1

    def fit(self, X_feature, y_train, isScaled = False):
        """
        Fit the SVM model to the training data.
        
        Parameters:
        - X_feature: array-like, shape (n_samples, n_features)
            The feature matrix for training.
        - y_train: array-like, shape (n_samples,)
            The target labels for training.
        """
        X_feature = np.array(X_feature)
        X_feature = X_feature *self.scaler if not isScaled else X_feature
        # Fit the model to the data
        self.svm_clf.fit(X_feature, y_train)

    def predict(self, points, isScaled = False):
        """
        Predict the labels of the given latent points using the trained SVM model.
        
        Parameters:
        - latent_points: array-like, shape (n_samples, n_features)
            The points to be classified.
            
        Returns:
        - predictions: array, shape (n_samples,)
            Predicted class labels for the input samples.
        """
        points = np.array(points)
        points = points *self.scaler if not isScaled else points

        # Predict the class labels for the given points
        return self.svm_clf.predict(points)
    
    def distance_to_boundary(self, point, isScaled = False):
        """
        Calculate the distance of a point from the decision boundary.
        """
        point = np.array(point).reshape(1, -1)
        point = point *self.scaler if not isScaled else point

        decision_value = self.svm_clf.decision_function(point)
        
        # distance = np.abs(decision_value) / np.linalg.norm(self.svm_clf.dual_coef_)
        
        # return distance[0]
        return np.abs(decision_value[0])

    def direction_to_boundary(self, point, epsilon=1e-3,  isScaled = False):
        """
        Calculate the direction vector towards the decision boundary.
        
        Parameters:
        - point: array-like, shape (n_features,)
            The point for which to calculate the direction to the decision boundary.
        - epsilon: float, optional (default=1e-5)
            A small perturbation to approximate the gradient.
            
        Returns:
        - direction: array, shape (n_features,)
            The normalized direction vector pointing towards the decision boundary.
        """
        point = np.array(point).reshape(1, -1)

        point = point *self.scaler if not isScaled else point

        # Perturb the point slightly to calculate the numerical gradient
        perturbed_point_positive = point + epsilon
        perturbed_point_negative = point - epsilon

        # Calculate the decision function for the perturbed points
        decision_value_positive = self.svm_clf.decision_function(perturbed_point_positive)[0]
        decision_value_negative = self.svm_clf.decision_function(perturbed_point_negative)[0]

        # Approximate the gradient of the decision function using numerical differentiation
        gradient = (decision_value_positive - decision_value_negative) / (2 * epsilon)

        # Move opposite to the gradient to approach the decision boundary
        direction = -gradient  # Moving towards the decision boundary
        direction = direction / np.linalg.norm(direction)  # Normalize the direction

        return direction
    
    def plot_hyperplane_with_points(self, X_train, y_train, features,  isScaled = False,  elev=20, azim=30):
        X_train = X_train *self.scaler if not isScaled else X_train

        # Define the grid for the feature space (across all three features)
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        z_min, z_max = X_train[:, 2].min() - 1, X_train[:, 2].max() + 1

        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 50),
                                np.linspace(y_min, y_max, 50),
                                np.linspace(z_min, z_max, 50))

        # Evaluate the decision function over the grid
        grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        Z = self.svm_clf.decision_function(grid)
        Z = Z.reshape(xx.shape)

        tolerance = 0.1  # Adjust this value based on how strict you want to be
        boundary_points = np.abs(Z) < tolerance

        boundary_coords = grid[boundary_points.ravel()]

        # Create a 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the decision boundary surface (where decision function ≈ 0)
        ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], boundary_coords[:, 2], 
                color='gray', marker='.', alpha=0.7, label='Decision Boundary')

        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], 
                c='r', label='Usable', s=20)

        # Plot data points where y_train = 0
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], 
                c='b', label='Unusable', s=20)


        # Set labels and legend
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        ax.legend()
        ax.view_init(elev=elev, azim=azim)
        # Show the plot
        plt.show()


    def move_to_closest_valid_boundary(self, point, step_size=0.01, isScaled = False):
        if isScaled:
            point = np.array(point, dtype=np.float64)
        else:
            point = np.array(point, dtype=np.float64) *self.scaler
        
        # Calculate the direction to the boundary
        direction = self.direction_to_boundary(point, isScaled=True)
        # Normalize the direction vector to unit length
        direction = direction / np.linalg.norm(direction) 
        new_point = point - step_size * direction 

        if not isScaled:
            new_point = np.array(new_point, dtype=np.float64)  / self.scaler

        return new_point

        