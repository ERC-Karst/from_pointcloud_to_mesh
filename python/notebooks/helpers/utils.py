import numpy as np

def load_matrix_from_file(filepath):
    """
    loads a cloud compare transformation matrix text file and
    returns a string representation.

    ----------
    
    arguments:

        fp -> str: the filepath

    ----------
    
    returns :

        m_data_str -> str : the string representation of 4 x 4 matrix
    
    
    """
    with open(filepath) as f:
        matrix_data = " ".join(f.readlines())
        f.close()

    return matrix_data


def suggest_global_shift(targets):
    """
    calculates the centroid of the target coordinates and returns the nearest coordinate triplet
    rounded to the nearest 100 m

    ----------
    
    arguments:

        targets -> np.array: a numpy array with N target coordinates (N x 3 matrix)
    ----------
    
    returns :

        globalShift -> np.array : a numpy array containing the suggested X, Y and Z shifts. 
    
    """

    centroid = np.mean(targets.T, axis = 1).reshape((-1,1))
    globalShift = np.round(centroid/100,0) * 100

    return globalShift  

def computePolygonArea(x, y):
    """
    calculates the area of a closed polygon whose vertex i coordinates are given by x[i] and y[i]
    

    ----------
    
    arguments:

        x -> np.array: a numpy array with N x coordinates (N x 1 matrix)
        y -> np.array: a numpy array with N y coordinates (N x 1 matrix)

    ----------
    
    returns :

        area -> float : the area in input units squared 
    
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))