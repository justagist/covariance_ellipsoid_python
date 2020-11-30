import numpy as np
from scipy.stats import chi2
from covell_plot_utils import plot_2d_ellipse, plot_3d_ellipsoid

def get_mean_and_covariance_matrix(data, **kwargs):
    """
    Get mean and covariance of the provided data

    :param data: input datapoints. m x n matrix, containing n number of m-dimensional datapoints.
    :type data: np.ndarray
    :param kwargs: additional keyword arguments to be passed to np.cov() method
    """
    if len(data.shape) == 1:
        data = data.reshape([1,-1])
    
    return np.mean(data, axis=1), np.cov(data,**kwargs)

def get_covariance_ellipsoid(covariance, confidence=0.95, mean=None):
    """
    Get the parameters of the covariance ellipsoid of given confidence level: center, axis vectors. The axes correspond to the ordered Eigen vectors in order of increasing eigen values. Use first n columns of the returned matrix to get the n most 'prominent' directions of distribution. Note that the ellipsoid axis are in the frame centered at the mean.

    :param covariance: covariance matrix
    :type covariance: np.ndarray
    :param mean: mean of data (optional); defaults to zero
    :type mean: np.ndarray (shape: [n,])
    :param confidence: desired confidence level for ellipse; dictates scale of ellipse. Computed as the chi-squared percent point function; prob of (1-confidence).
    :type confidence: float
    :return center, axes of ellipsoid centered at mean
    :rtype: [np.ndarray, np.ndarray]
    """
    if mean is None:
        mean = np.zeros(covariance.shape[0])
    
    dof = mean.shape[0]
    
    w, v = np.linalg.eig(covariance)

    ## -- sort in decreasing order of eigenvalues
    idx = w.argsort()[::-1]   
    w = w[idx]
    v = v[:,idx]

    scale = chi2.ppf(confidence, dof)
    axes = np.sqrt(scale)*np.sqrt(w)*v

    return mean.flatten(), axes

def point_is_in_covariance_ellipsoid(point, mean, covariance, confidence=0.85):
    """
    Check if the provided point is in a covariance ellipsoid defined by the given mean, covariance and confidence interval.

    :param point: the point to check
    :type point: np.ndarray
    :param mean: center of ellipse
    :type mean: np.ndarray
    :param covariance: covariance matrix
    :type covariance: np.ndarray
    :param confidence: confidence interval to test against, defaults to 0.85
    :type confidence: float, optional
    :return: True if point is within the ellipse, False otherwise
    :rtype: bool
    """
    point = point.reshape([-1,1]).copy()
    mean = mean.reshape([-1,1]).copy()
    return (point - mean).T.dot(np.linalg.inv(covariance).dot(point-mean)) <= chi2.ppf(confidence, mean.shape[0])

def plot_covariance_ellipsoid(mean, covariance, confidence=0.95, datapoints=None, ax=None):
    """
    Draw a covariance ellipsoid of specified confidence.
        mean, covariance, and datapoints dimensions are assumed to be corresponding. Only first three (at most) dimensions will be plotted.

    :param mean: mean of data (shape: [n,])
    :type mean: np.ndarray 
    :param covariance: covariance matrix (shape [n,n])
    :type covariance: np.ndarray
    :param datapoints: data for plotting scattered points (shape: [n,num_points]), defaults to None 
    :type datapoints: np.ndarray, optional
    """

    dof = min(mean.shape[0],3)

    center, axes = get_covariance_ellipsoid(covariance, confidence, mean)

    if dof == 2:
        ax = plot_2d_ellipse(center,axes,ax)
    elif dof == 3:
        ax = plot_3d_ellipsoid(center, axes, ax)
    else:
        return False

    for i in range(dof):
        ax.plot(*[[center[d],axes[d,i]+center[d]] for d in range(dof)], c='r')
    
    if datapoints is not None:
        ax.scatter(*[datapoints[i,:] for i in range(dof)])
    
    return ax

def plot_covariance_ellipsoid_from_data(data, confidence=0.95, ax=None):
    mean, cov = get_mean_and_covariance_matrix(data)
    ax = plot_covariance_ellipsoid(mean, cov, confidence=confidence, datapoints=data, ax=ax)
    return ax

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ax = None
    
    ## -- Test 3D data
    s = np.asarray([2,2,2]).reshape([3,1])
    x = np.random.randn(334)
    y = np.random.standard_normal([3,334])+s*x.reshape([1,-1])+10

    ## -- Test 2D data
    # s = np.asarray([2,2]).reshape([2,1])
    # x = np.random.randn(334)
    # y = np.random.standard_normal([2,334])+s*x.reshape([1,-1])+10

    ax = plot_covariance_ellipsoid_from_data(y,confidence=0.95, ax=ax)

    try:
        plt.axis("equal")
    except NotImplementedError:
        plt.axis("auto")
    # ax.set_xlim([-7,7])
    # ax.set_ylim([-7,7])
    # if y.shape[0] > 2:
    #     ax.set_zlim([-7,7])
    plt.show()
