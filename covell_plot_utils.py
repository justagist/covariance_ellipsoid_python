import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

def plot_2d_ellipse(center, axes, ax=None):
    """
    plot 2d ellipse
    axes should be specified with respect to the center.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    unit_vector_1 = axes[:,0] / np.linalg.norm(axes[:,0])
    dot_product = np.dot(unit_vector_1, [1,0])
    angle = np.arccos(dot_product)*np.sign(unit_vector_1[1])
    
    ellipse = Ellipse(center, width=2*np.linalg.norm(axes[:,0]), height=2*np.linalg.norm(axes[:,1]),
                      angle=angle*180/np.pi, alpha=0.2)

    ax.add_patch(ellipse)
    return ax

def plot_3d_ellipsoid(center, axes, ax=None):
    """
    plot 3d ellipse
    axes should be specified with respect to the center.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    x = np.linalg.norm(axes[:,0]) * np.outer(np.cos(u), np.sin(v))
    y = np.linalg.norm(axes[:,1]) * np.outer(np.sin(u), np.sin(v))
    z = np.linalg.norm(axes[:,2]) * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot(axes/np.linalg.norm(axes,axis=0),[x[i,j],y[i,j],z[i,j]]) + center

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    return ax