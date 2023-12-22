import numpy as np
from scipy.spatial import ConvexHull


class Section:

    def __init__(self, points, flag, from3d = True):
        self.points = points
        self.flag = flag
        self.from3d = from3d

    def compute_descriptors(self):
        if self.from3d == True:
            self.points2d = self.points[:, [0, 2]]
        else: 
            self.points2d = self.points

        self.perimeter = computePolygonPerimeter(self.points2d)
        self.area = computePolygonArea(self.points2d)
        self.convexhull = computeConvexHull(self.points2d)
        self.convexhull_area = computePolygonArea(self.convexhull)
        self.hydraulic_diameter = computeHydraulicDiameter(self.area, self.perimeter)
        self.solidity = self.area / self.convexhull_area
        self.circularity = 4 * np.pi * self.area / self.perimeter**2


    def print_basic_stats(self):
        print("Basic stats for the chosen section")
        print("------------------------------------\n")
        print(f"Perimeter : {self.perimeter:5.2f}m")
        print(f"Area : {self.area:5.2f} m2")
        print(f"Hydraulic diameter : {self.hydraulic_diameter:5.2f}m")
        print(f"Solidity : {self.solidity:5.2f}")
        print(f"Circularity : {self.circularity:5.2f}\n")




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
    