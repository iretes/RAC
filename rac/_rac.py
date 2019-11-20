"""
Implementation of the RAC classifier
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import rankdata


class RacClassifier(BaseEstimator, ClassifierMixin):
    """ RAC Classifier class.

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
    
    def __init__(self, weighted=False):
        self.weighted = weighted

    def fit(self, X, y):
        """Computes class signatures uring rank aggregation. 
        The features are ranked for each training sample and the Borda count algorithm
	is applied to all seamples from each class.

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
        self.class_signatures_=np.empty((len(self.classes_),X.shape[1]))
        for i in range(len(self.classes_)):
            self.class_signatures_[i]=self.aggregate(self.X_[self.y_==self.classes_[i]])
	
        # Return the classifier
        return self

    def predict(self, X):
        """ Predicts the class for each input sample. 
	The prediction is based on the distance between the sample and the class signatures. 
	The distance is computed using Spearman's distance between rankings, weighted by the rank if the weighted parameter is set.

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

        closest = np.argmin(self. distances_to_signatures(X), axis=1)
        return self.classes_[closest]
    
    def distances_to_signatures(self,X):
        """Computes the matrix of pairwise Spearman distances between the ranking of features in the two sample sets.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input samples to compare to signatures.

        Returns
        -------
        d : array-like, shape (n_samples, n_classes)
            The distance matrix between all samples from X to all class signatures.
        """
        if 0 in X.shape:
           return np.empty(0)
        weights=1
        if self.weighted:
            weights=np.abs(X.shape[1]-1-2*(self.class_signatures_-1))
        d=np.empty((X.shape[0],len(self.classes_)))
        for i in range(X.shape[0]):
            d[i]=np.sum(np.abs(self.class_signatures_-rankdata(X[i], method='min'))*weights,axis=1)
        return d
            
        
    def aggregate(self, X):
        """Ranks features and aggregates them into one signature.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples from one class only.

        Returns
        -------
        s : ndarray, shape (n_features,)
            The signature determined by the input samples. 
            Smallest rank corresponds to smallest feature value.
        """
        s=np.zeros(X.shape[1])
        for sample in X:
            s=s+rankdata(sample, method='min')
        return rankdata(s,method='min')
    
    def get_params(self, deep=True):
        return {"weighted": self.weighted}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self