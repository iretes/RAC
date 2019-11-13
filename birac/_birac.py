"""
Implementation of the BiRAC classifier
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels



class BiRacClassifier(BaseEstimator, ClassifierMixin):
    """ BiRAC Classifier class.

    Parameters
    ----------
    weighted : Boolean, default=False
        Parameter that specifies whether the Spearman distance to the signatures is weighted by the rank or not.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    class_signatures_ : ndarray, shape (n_classes, n_features)
        The signatures for each class, computed during :meth:`fit`.
    """
    
    def __init__(self, weighted=True):
        self.weighted = weighted

    def fit(self, X, y):
        """Computes class signatures uring rank aggregation. The features are ranked for each training sample and the Borda count algorithm is applied to all seamples from each class.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

	#compute signature for each class
	
	
        # Return the classifier
        return self

    def predict(self, X):
        """ Predicts the class for each input sample. The prediction is based on the distance between the sample and the class signatures. The distance is computed using Spearman's distance between rankings, weighted by the rank if the weighted parameter is set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
