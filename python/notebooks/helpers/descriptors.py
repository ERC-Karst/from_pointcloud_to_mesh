import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from scipy.optimize import minimize 

class Section:

    def __init__(self, points, flag, curvilinear_pos, from3d = True):
        self.points = points
        self.flag = flag
        self.from3d = from3d
        self.curvilinear_pos = curvilinear_pos

        self.offset = None
        self.points2d = None
        self.perimeter = None
        self.area = None
        self.convexhull = None
        self.convexhull_area = None
        self.hydraulic_diameter = None
        self.hydraulic_radius = None
        self.solidity = None
        self.circularity = None
        self.mean_scaled_rh_deviation = None
        self.fitted_ellipse = None
        self.mean_dist_to_ellipse_scaled = None

    def compute_descriptors(self):
        if self.from3d == True:
            pts = self.points[:, [0, 2]]

        else: 
            pts = self.points
        
        self.offset = np.mean(pts, axis = 0)
        self.points2d = pts - self.offset
        self.perimeter = computePolygonPerimeter(self.points2d)
        self.area = computePolygonArea(self.points2d)
        self.convexhull = computeConvexHull(self.points2d)
        self.convexhull_area = computePolygonArea(self.convexhull)
        self.hydraulic_diameter = computeHydraulicDiameter(self.area, self.perimeter)
        self.hydraulic_radius = self.hydraulic_diameter / 2
        self.solidity = self.area / self.convexhull_area
        self.circularity = 4 * np.pi * self.area / self.perimeter**2
        self.mean_scaled_rh_deviation = np.mean( 1 / (2 * self.hydraulic_diameter) * (np.linalg.norm(self.points2d, axis =1) - 1))

    def fitEllipse(self):
        # first part is PCA transform of the shape. 
        X = self.points2d
        X_mean = X.mean(axis = 0)
        X_std = X.std(axis = 0)
        Z = (X - X_mean) / X_std
        COV = Z.T @ Z
        eigenvalues, eigenvectors = np.linalg.eig(COV)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx[::-1]]
        eigenvectors = eigenvectors[:,idx[::-1]]
        D = np.diag(eigenvalues)
        P = eigenvectors
        Z_new = np.dot(Z, P)

        # minimize the sum of square distances. 
        out = minimize(sumSquaredDistToEllipse, [1,1], Z_new)

        self.fitted_ellipse = makeEllipse((0,0), out.x[0], out.x[1], 100, 0) @ np.linalg.inv(P) * X_std + X_mean
        semi_major = np.array((out.x[0], 0)) @ np.linalg.inv(P) * X_std + X_mean
        semi_minor = np.array((0, out.x[1])) @ np.linalg.inv(P) * X_std + X_mean
        self.ellipse_axes = (np.linalg.norm(semi_major), np.linalg.norm(semi_minor))
        # compute the average distance between shape and transformed / rescale ellipse. / scale by hydraulic radius-
        self.mean_dist_to_ellipse= np.mean(np.min(np.linalg.norm(np.expand_dims(self.fitted_ellipse, 1) - self.points2d, axis = 2).T, axis = 1)) 
        self.mean_dist_to_ellipse_scaled =  self.mean_dist_to_ellipse / self.hydraulic_radius

    
    def print_basic_stats(self):
        print("Basic stats for the chosen section")
        print("------------------------------------\n")
        print(f"Perimeter : {self.perimeter:5.2f}m")
        print(f"Area : {self.area:5.2f} m2")
        print(f"Hydraulic diameter : {self.hydraulic_diameter:5.2f}m")
        print(f"Solidity : {self.solidity:5.2f}")
        print(f"Circularity : {self.circularity:5.2f}\n")
        print(f"Mean scaled distance to hydraulic radius : {self.mean_scaled_rh_deviation:5.2f}\n")


    def plot_basic(self, ax, maxdim =5, orientation = "horizontal"):
        """
        plots the section, and a barycentred disk with the section's hydraulic radius
        """

        thetai = np.linspace(-np.pi, np.pi, 100)

        # centroid
        ax.scatter(0, 0)

        # actual section
        ax.plot(self.points2d[:, 0], self.points2d[:, 1], zorder = 50, color = "k", label = "original")
        
        # ellipse
        ax.plot(self.fitted_ellipse[:, 0], self.fitted_ellipse[:, 1], zorder = 100, color = "r", label = "fitted ellipse")

        # circle of hydraulic diameter, centered
        ax.fill_between(np.cos(thetai)*self.hydraulic_diameter/2, 
                        np.sin(thetai)*self.hydraulic_diameter/2, 0, facecolor="dodgerblue", alpha = 0.5)

        # plot convex hull
        chull = self.convexhull
        chull_wrapped = np.zeros((len(chull)+1, 2))
        chull_wrapped[:-1] = chull
        chull_wrapped[-1] = chull[0]
        ax.plot(chull_wrapped[:,0], chull_wrapped[:, 1], zorder = -10, color = "C1")

        # fix axes
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal");

        # info box
        stats_str = f"""
        Area: {self.area:18.1f} m$^2$
        $R_h$: {self.hydraulic_radius:21.1f} m
        circularity: {self.circularity:10.2f}
        solidity: {self.solidity:14.2f}
        scaled avg. dist. to ellipse: {self.mean_dist_to_ellipse_scaled:4.2f}
        """
        bbox = dict(boxstyle="square,pad=0.1", fc="None", ec="None")

        if orientation == "horizontal":  
            ax.annotate(stats_str,(0,0),(-1.2 * maxdim,maxdim *1.5),  bbox=bbox, fontsize = 14, va = "center")
        else: 
            ax.annotate(stats_str,(0,0),(maxdim *1.1, -0.5 * maxdim,),  bbox=bbox, fontsize = 14, va = "center")

        
        return ax
        
def computePolygonPerimeter(a):
    """
    calculates the perimeter of a closed polygon whose vertex i coordinates are given by x[i] and y[i]
    

    ----------
    
    arguments:

        a -> np.array: a numpy array with N coordinates (N x 2 matrix)

    ----------
    
    returns :

        perimeter -> float : the area in input units squared 
    """
    return np.sqrt(np.sum((np.roll(a,1, axis = 0) - a)**2, axis = 1)).sum()

def computePolygonArea(a):
    """
    calculates the area of a closed polygon whose vertex i coordinates are given by x[i] and y[i]
    

    ----------
    
    arguments:

        a -> np.array: a numpy array with N coordinates (N x 2 matrix)

    ----------
    
    returns :

        area -> float : the area in input units squared 
    
    """
    x, y = a.T
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def computeHydraulicDiameter(a, p):
    """
    calculates the area of a closed polygon whose vertex i coordinates are given by x[i] and y[i]
    

    ----------
    
    arguments:

        a -> scalar: area of a section
        p -> scalar: perimeter of a section 

    ----------
    
    returns :

        hydraulic diameter -> float : the area in input units squared 
    """
    
    return 4 * a / p


def computeConvexHull(a):
    hull = ConvexHull(a)

    hull_vertices = np.array((a[hull.vertices, 0], a[hull.vertices, 1])).T

    return hull_vertices

def makeEllipse(xy, a, b, numpoints = 100, shear = 0):
    centre = np.array(xy)
    t = np.linspace(-np.pi, np.pi, numpoints)
    # cartesian coordinates of the ellipse
    ellipse_x = a* np.cos(t)
    ellipse_y = b*np.sin(t)
    ellipse = np.vstack([ellipse_x, ellipse_y]).T + centre
    # define simple shear matrix
    simple_shear = np.array([[1, shear],[0, 1]])
    # apply if needed, by default, Identity matrix 
    return ellipse @ simple_shear

def sumSquaredDistToEllipse(p, pts):
    """
    calculates the sum of squared distances between a given shape and an ellipse of given parameters p
    

    ----------
    
    arguments:
        p -> tuple: (a, b) where a and b are the semi-axes of an ellipse.
        pts -> array (N, 2) coordinates of the shape vertices.
    ----------
    
    returns :

        sum_squared_distances -> float : the area in input units squared 
    """
    a, b = p
    # find the nearest distance to a given point on an ellipse.
    ell = makeEllipse((0,0), a, b, 100, 0)

    abs_dist = np.min(np.linalg.norm(np.expand_dims(ell, 1) - pts, axis = 2).T, axis = 1)
    sum_squared_distances = np.sum(abs_dist**2)
    
    return sum_squared_distances    