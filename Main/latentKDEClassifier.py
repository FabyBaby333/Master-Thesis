import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)


class LatentKDEClassifier:
    def __init__(self, bandwidt_usable:int,bandwidt_unusable:int, min_confidence_usable= None):
        self.kde_usable = KernelDensity(kernel='gaussian', bandwidth=bandwidt_usable)
        self.kde_unusable = KernelDensity(kernel='gaussian', bandwidth=bandwidt_unusable)
        self.min_confidence = min_confidence_usable

    def set_min_usable_confidence(self,min_confidence_usable = None):
        self.min_confidence = min_confidence_usable

    def fit(self, latent_usable_train, latent_unusable_train):
        """
        Fit the KDE models to the usable and unusable training data.
        """
        self.kde_usable.fit(latent_usable_train)
        self.kde_unusable.fit(latent_unusable_train)

    def get_log_odds(self,latent_points) :
        return  self.kde_usable.score_samples(latent_points), self.kde_unusable.score_samples(latent_points)
                     
    def predict(self, latent_points):
        """
        Predict class labels for the given latent points.
        Returns an array of predicted labels (1 for usable, 0 for unusable).
        """
        # Ensure latent_points is a 2D array
        if latent_points.ndim == 1:
            latent_points = latent_points.reshape(1, -1)

        # Compute log densities for usable and unusable
        log_density_usable, log_density_unusable = self.get_log_odds(latent_points)
        
        # print(f'log_likelihood_usable={log_density_usable}  log_likelihood_unusable= {log_density_unusable}')

        # Predict based on which log density is higher
        predictions = (log_density_usable > log_density_unusable) & (
            log_density_usable > self.min_confidence if self.min_confidence is not None else True
            ) 

        # Convert boolean array to integer array (1 for usable, 0 for unusable)
        return predictions.astype(int)
    
    def log_likelihoods(self, point):
        point = np.reshape(point, (1, -1))  # Make sure the point is in the right shape
        return  self.kde_usable.score_samples(point), self.kde_unusable.score_samples(point)

    def calculate_log_odds(self,dataset):
        log_odds = []
        for point in dataset:
            log_likelihood_usable, log_likelihood_unusable = self.log_likelihoods(point)
            log_odds_point = log_likelihood_usable - log_likelihood_unusable
            log_odds.append(log_odds_point)
        return log_odds
    
    def compute_gradient(self, point, epsilon = 1e-3):
        num_features = point.shape[0]
        gradient_usable = np.zeros(num_features)
        gradient_unusable = np.zeros(num_features)
        
        for i in range(num_features):
            # Perturb the point slightly in positive and negative direction
            point_plus = np.copy(point)
            point_minus = np.copy(point)
            point_plus[i] += epsilon
            point_minus[i] -= epsilon
            
            # Compute log-likelihoods for perturbed points
            log_likelihood_usable_plus, log_likelihood_unusable_plus = self.log_likelihoods(point_plus)
            log_likelihood_usable_minus, log_likelihood_unusable_minus = self.log_likelihoods(point_minus)
            
            
            # Approximate gradients
            gradient_usable[i] = (log_likelihood_usable_plus[0] - log_likelihood_usable_minus[0]) / (2 * epsilon)
            gradient_unusable[i] = (log_likelihood_unusable_plus[0] - log_likelihood_unusable_minus[0]) / (2 * epsilon)
        
        return gradient_usable, gradient_unusable


    def _get_direction(self, log_likelihood_usable,  log_likelihood_unusable,gradient_usable, gradient_unusable, use_direction = 0, alpha = 0.2):

        norm_gradient_usable = np.linalg.norm(gradient_usable)
        if norm_gradient_usable != 0: 
            gradient_usable /= norm_gradient_usable

        norm_gradient_unusable = np.linalg.norm(gradient_unusable)
        if norm_gradient_unusable != 0: 
            gradient_unusable /= norm_gradient_unusable

        if use_direction == 3:
            return gradient_usable - gradient_unusable
        elif use_direction == 1:
            return gradient_usable
        elif use_direction == 2:
            return -gradient_unusable
        elif use_direction == 5 :
            if log_likelihood_usable > log_likelihood_unusable:
                return gradient_usable - alpha*gradient_unusable
            else:
                return -gradient_unusable + alpha*gradient_usable
        elif use_direction == 4 :
            if log_likelihood_usable > log_likelihood_unusable:
                return gradient_usable 
            else:
                return -gradient_unusable
            
            
    def move_point(self, point, step_scale=0.1, use_direction = 3, alpha = None, target: bool=True):
        """
        Adjusts the given point by moving them in the direction determined by the gradients of log-likelihoods.
        
        Parameters:
            points (array-like): The input point to be adjusted.
            step_scale (float): The step size for moving the points.
            use_direction (int): Determines the strategy for combining gradients.
            alpha (float): The weight factor for balancing usable and unusable gradients.
            target (bool): True to move towards usable, False for unusable
        Returns:
            np.ndarray: The adjusted points.
        """
        log_likelihood_usable, log_likelihood_unusable = self.log_likelihoods(point)
        print(f'log_likelihood_usable={log_likelihood_usable}  log_likelihood_unusable= {log_likelihood_unusable}')

        gradient_usable, gradient_unusable = self.compute_gradient(point, step_scale)
        
        gradient = self._get_direction(log_likelihood_usable,  log_likelihood_unusable,gradient_usable, gradient_unusable, use_direction, alpha)

        norm = np.linalg.norm(gradient)
        if norm != 0: 
            gradient /= norm
        
        if target:
            point += step_scale * gradient
        else:
            point += step_scale * (-gradient)
    
        return np.array(point)
    


    def move_points(self, points, step_scale=0.1, use_direction=3, alpha=None, max_iter=1, verbose=False):
        """
        Adjusts the given points by moving them in the direction determined by the gradients of log-likelihoods.

        Parameters:
            points (array-like): The input points to be adjusted.
            step_scale (float): The step size for moving the points.
            use_direction (int): Determines the strategy for combining gradients.
            alpha (float): The weight factor for balancing usable and unusable gradients.
            max_iter (int): Maximum number of iterations for adjustment.
            verbose (bool): If True, prints detailed logs for insights.

        Returns:
            np.ndarray: The adjusted points.
        """
        new_points = []

        for i, point in enumerate(points):
            current_point = np.copy(point)
            converged = False

            if verbose:
                print(f"Point {i}: Starting adjustment")

            for step in range(max_iter):
                if self.predict(current_point):
                    converged = True
                    break

                log_likelihood_usable, log_likelihood_unusable = self.log_likelihoods(current_point)
                gradient_usable, gradient_unusable = self.compute_gradient(current_point, step_scale)
                gradient = self._get_direction(
                    log_likelihood_usable,
                    log_likelihood_unusable,
                    gradient_usable,
                    gradient_unusable,
                    use_direction,
                    alpha,
                )

                norm = np.linalg.norm(gradient)
                if norm != 0:
                    gradient /= norm

                current_point += step_scale * gradient

                if verbose:
                    print(
                        f"Point {i}, Step {step}: log_likelihood_usable={log_likelihood_usable}, "
                        f"log_likelihood_unusable={log_likelihood_unusable}"
                    )

            # Log results for the final state
            if not converged and verbose:
                log_likelihood_usable, log_likelihood_unusable = self.log_likelihoods(current_point)
                print(
                    f"Point {i}: Did not converge within {max_iter} steps, final log_likelihoods - "
                    f"usable={log_likelihood_usable}, unusable={log_likelihood_unusable}"
                )

            if converged and verbose:
                log_likelihood_usable, log_likelihood_unusable = self.log_likelihoods(current_point)
                print(f"Point {i}: Adjustment complete after {step} steps, final log_likelihoods - "
                    f"usable={log_likelihood_usable}, unusable={log_likelihood_unusable}"
                    )

            new_points.append(current_point)

        return np.array(new_points)




