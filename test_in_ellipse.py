from covariance_ellipsoid import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    ax = None
    conf = 0.75
    
    ## -- Test 3D data
    s = np.asarray([2,2,2]).reshape([3,1])
    x = np.random.randn(334)
    y = np.random.standard_normal([3,334])+s*x.reshape([1,-1])+10

    ## -- Test 2D data
    # s = np.asarray([2,2]).reshape([2,1])
    # x = np.random.randn(334)
    # y = np.random.standard_normal([2,334])+s*x.reshape([1,-1])+10

    mean, cov = get_mean_and_covariance_matrix(y)
    
    center, axes = get_covariance_ellipsoid(cov, conf, mean)

    in_points = []
    out_points = []
    
    ax = plot_covariance_ellipsoid(mean, cov, conf)

    for i in range(y.shape[1]):
        # print y[:,i]
        if point_is_in_covariance_ellipsoid(y[:,i], mean, cov, conf):
            in_points.append(y[:,i].flatten())
            # print "yes"
        else:
            out_points.append(y[:,i].flatten())
            
    in_points = np.asarray(in_points)
    out_points = np.asarray(out_points)
    
    if in_points.shape[0]>1:
        ax.scatter(*[in_points[:,d] for d in range(y.shape[0])],c='b')
    if out_points.shape[0]>1:
        ax.scatter(*[out_points[:,d] for d in range(y.shape[0])],c='r')
    try:
        ax.axis("equal")
    except NotImplementedError:
        ax.axis("auto")
        
    plt.show()