import numpy as np
from scipy.spatial import ConvexHull
from polylabel import polylabel
import time

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
        self.centre_of_gravity = centre_of_gravity(self.points2d)
        self.perimeter = computePolygonPerimeter(self.points2d)
        self.area = computePolygonArea(self.points2d)
        self.convexhull = computeConvexHull(self.points2d)
        self.convexhull_area = computePolygonArea(self.convexhull)
        self.hydraulic_diameter = computeHydraulicDiameter(self.area, self.perimeter)
        self.hydraulic_radius = self.hydraulic_diameter / 2
        self.solidity = self.area / self.convexhull_area
        self.circularity = 4 * np.pi * self.area / self.perimeter**2
        self.mean_scaled_rh_deviation = np.mean( 1 / (2 * self.hydraulic_diameter) * (np.linalg.norm(self.points2d, axis =1) - 1))
        #self.inaccessible_pole, self.inscribed_circle_radius = polylabel([self.points2d.tolist()], with_distance=True, precision= 1e-3)

    def fitEllipse(self, with_minimize = False):
        # first part is PCA transform of the shape. 
        X = self.points2d
        G = self.centre_of_gravity
        Sigma = 1/ self.area *  X.T @ X
        D, S = np.linalg.eig(Sigma)
        self.aspect_ratio = D[0] / D[1]
        idx = np.argsort(D)
        D = np.diag(D)
        X_prime = S.T @ X.T
        _lambda = np.sqrt(self.area  / (np.sqrt(np.linalg.det(Sigma))*np.pi))
        # determined ellipse
        t = np.linspace(0,2*np.pi,500)
        Y = np.vstack((np.cos(t), np.sin(t)))
        self.determined_ellipse = (_lambda * S @ np.sqrt(D) @ Y).T + G

        if with_minimize == True:
            # minimize the sum of square distances. 
            out = minimize(objective_ellipse, [_lambda * np.sqrt(D[0,0]),_lambda * np.sqrt(D[1,1]), 0], X_prime.T, method = "Nelder-Mead")
            Y2 = makeEllipse(*out.x)
            self.fitted_ellipse = (S @ Y2.T).T + G

            self.ellipse_axes = (out.x[0], out.x[1])

        
            # compute the average distance between shape and transformed / rescale ellipse. / scale by hydraulic radius-
            self.dist_to_ellipse = np.min(np.linalg.norm(np.expand_dims(self.fitted_ellipse, 1) - self.points2d, axis = 2).T, axis = 1)
            self.average_directed_distance_to_ellipse = directed_distance(self.points2d, self.fitted_ellipse)
    
            self.dist_vectors_arguments= np.argmin(np.linalg.norm(np.expand_dims(self.fitted_ellipse, 1) - self.points2d, axis = 2).T, axis = 1)
    
            self.dist_vectors = (np.expand_dims(self.fitted_ellipse, 1) - self.points2d)
    
            self.vectors_to_ellipse = np.vstack([self.dist_vectors[self.dist_vectors_arguments[i], i] for i in range(len(self.dist_vectors_arguments))])
    
            self.outward = np.array([self.vectors_to_ellipse[i].dot(self.points2d[i]) > 0 for i in range(len(self.points2d))])
    
            # calculate the distance to centroid: 
            centroid_dist = np.linalg.norm(X_prime, axis=1)
            
            thetai = np.arctan2(X_prime[:,1], X_prime[:,0])
    
            ellipse_boundary = np.linalg.norm(np.vstack((out.x[0]*np.cos(thetai), out.x[1]*np.sin(thetai))).T, axis = 1)
    
            self.mean_dist_to_ellipse = np.mean(self.dist_to_ellipse)
            self.mean_dist_to_ellipse_scaled =  self.mean_dist_to_ellipse / self.hydraulic_radius
        self.ellipse_area = computePolygonArea(self.determined_ellipse)
        self.ellipse_perimeter = computePolygonPerimeter(self.determined_ellipse)
    
    def print_basic_stats(self):
        print("Basic stats for the chosen section")
        print("------------------------------------\n")
        print(f"Perimeter : {self.perimeter:5.2f}m")
        print(f"Area : {self.area:5.2f} m2")
        print(f"Hydraulic diameter : {self.hydraulic_diameter:5.2f}m")
        print(f"Solidity : {self.solidity:5.2f}")
        print(f"Circularity : {self.circularity:5.2f}\n")
        print(f"Mean distance to best fit ellipse : {self.mean_dist_to_ellipse:5.2f}\n")


    def plot_basic(self, ax, maxdim =5, orientation = "horizontal", verbose = True):
        """
        plots the section, and a barycentred disk with the section's hydraulic radius
        """

        thetai = np.linspace(-np.pi, np.pi, 100)

        # centroid
        ax.scatter(0, 0)

        # actual section
        ax.plot(self.points2d[:, 0], self.points2d[:, 1], zorder = 50, color = "k", label = "original")
        
        # ellipse
        #ax.plot(self.fitted_ellipse[:, 0], self.fitted_ellipse[:, 1], zorder = 100, color = "r", label = "fitted ellipse")
        ax.plot(self.determined_ellipse[:, 0], self.determined_ellipse[:, 1], zorder = 100, color = "g", label = "determined ellipse")

        # circle of hydraulic diameter, centered
        ax.plot(np.cos(thetai)*self.hydraulic_diameter/2, 
                        np.sin(thetai)*self.hydraulic_diameter/2,  color="dodgerblue", label = "hydraulic diameter circle\non centroid")

        # plot convex hull
        chull = self.convexhull
        chull_wrapped = np.zeros((len(chull)+1, 2))
        chull_wrapped[:-1] = chull
        chull_wrapped[-1] = chull[0]
        ax.plot(chull_wrapped[:,0], chull_wrapped[:, 1], zorder = -10, color = "C1", label = "section convex hull")

        # fix axes
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal");

        if verbose == True:
            # info box
            stats_str = f"""
            Area: {self.area:18.1f} m$^2$
            Area of fitted ellipse: {self.ellipse_area:10.2f} m$^2$
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

def makeEllipse(a, b, theta):
    # define ellipse parametrically
    t = np.linspace(-np.pi, np.pi, 500)
    # calculate coordinates before rotation
    xy = np.vstack([a*np.cos(t), b*np.sin(t)]).T
    #rotation matrix
    rot = np.array([[np.cos(theta),-np.sin(theta)],
                      [np.sin(theta),np.cos(theta)]]).astype(np.float32)
    # return rotated ellipse. 
    return np.dot(xy, rot)

def objective_ellipse(p, pts):
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
    a, b, theta = p
    # find the nearest distance to a given point on an ellipse.
    ellipse = makeEllipse( a, b, theta)

    #abs_dist = np.min(np.linalg.norm(np.expand_dims(ell, 1) - pts, axis = 2).T, axis = 1)
    #sum_squared_distances = np.sum(abs_dist**2)
    return average_directed_distance(ellipse, pts)
    

def directed_distance(S1, S2):
    S1toS2 = np.min(np.linalg.norm(np.expand_dims(S2, 1) - S1, axis = 2).T, axis = 1)
    return np.mean(S1toS2)
    
def average_directed_distance(S1, S2):
    S1toS2 = directed_distance(S1, S2)
    S2toS1 = directed_distance(S2, S1)
    return (S1toS2 + S2toS1) / 2


def centre_of_gravity(ngon, is3d=False, verbose = False):
    """
    computes the centre of gravity of a 2d or 3d n-gon

    ----------
    
    arguments:

        points -> np.array: a numpy array with N target coordinates (N x 3 matrix)
    ----------
    
    returns :

        cog -> np.array : a numpy array containing the coordinates of the centre of gravity (1 x 2 matrix)
    
    """
    n = len(ngon)
    # where ngon is the path determining the section by consecutive pairs of coordinates. 
    denominator = np.zeros(n-2)
    numerator = np.zeros((n-2, 2))
    
    # determinant method (see: https://math.stackexchange.com/questions/90463/how-can-i-calculate-the-centroid-of-polygon)
    u = ngon[1:-1] - ngon[0]
    v = ngon[2:] - ngon[0]
    
    dets = u[:, 0]* v[:, 1] - u[:, 1]*v[:, 0]
    centroids = (ngon[0]+ngon[1:-1]+ngon[2:]) / 3
    numerator = np.sum(np.multiply(np.expand_dims(dets, -1), centroids), axis = 0) 
    denominator = np.sum(dets)

    cog = numerator / denominator
    
    if verbose == True:
        print("estimated area",  np.abs(np.sum(u[:, 0]* v[:, 1] - u[:, 1]*v[:, 0]))/ 2)
    return cog