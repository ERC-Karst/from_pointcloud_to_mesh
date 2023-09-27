## Based on Arun et al., 1987 Least-Squares Fitting of Two 3-D Point Sets
## https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
    

import numpy as np
from typing import List

def pairWiseRegistration(p1_t, p2_t , verbose : str = False, atol = 1e-1) -> any:
    """
    Function to find the rotation and translation matrices for the rigid registration of point clouds.
    This  minimises the vector N for: 

    P' = RP + t1.T
    P' and P are 3 x N matrices representing two sets of N points to be registered. 
    R is the rotation matrix. 
    t is a translation vector. 
    1 is N by 1 vector of ones. 

    ----------
    
    arguments:

    p1_t : an N x 3 matrix of reference point coordinates 
    p2_t : an N x 3 matrix of target point coordinates
    atol : tolerance for the distance of references to targets given the transformation

    ----------
    
    returns:

    result : the target point coordinates newly registered using the rotation and translation matrices. 
    R : the rotation matrix. 
    t : the translation vector. 
    """
    
    p1 = p1_t.transpose()
    p2 = p2_t.transpose()

    #Calculate centroids
    p1_c = np.mean(p1, axis = 1).reshape((-1,1)) #If you don't put reshape then the outcome is 1D with no rows/colums and is interpeted as rowvector in next minus operation, while it should be a column vector
    p2_c = np.mean(p2, axis = 1).reshape((-1,1))

    if verbose == True:
        print("centroids:\n")
        print("to be aligned:\n", p1_c)
        print("reference cloud:\n", p2_c)
        
    #Subtract centroids
    q1 = p1-p1_c
    q2 = p2-p2_c

    #Calculate covariance matrix
    H=np.matmul(q1,q2.transpose())

    if verbose == True:
        print("covariance matrix H:\n", H)


    #Calculate singular value decomposition (SVD)
    U, X, V_t = np.linalg.svd(H) #the SVD of linalg gives you Vt

    if verbose == True:
        print("SVD of H:\n", V_t)
        print("X:\n", X)
        
    #Calculate rotation matrix
    R = np.matmul(V_t.transpose(),U.transpose())

    if verbose == True:
        print("Rotation matrix:\n", R)
        
    assert np.allclose(np.linalg.det(R), 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    #Calculate translation matrix
    T = p2_c - np.matmul(R,p1_c)
    if verbose == True:
        print("Translation matrix:\n", T)
    #Check result
    result = T + np.matmul(R,p1)
    if np.allclose(result,p2,atol=atol):
        print("transformation is correct!")
    else:
        print("transformation is wrong...")

    return result,R,T