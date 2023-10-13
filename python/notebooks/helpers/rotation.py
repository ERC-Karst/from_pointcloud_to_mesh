import numpy as np

def Householder(u):
    u = np.expand_dims(u, -1)
    return np.identity(u.shape[0]) - 2 *  (u @ u.T) / np.dot(u.T, u)

def RotateFromTo(f, t, verbose = False):

    """
    Rotates a vector f onto vector T by a single rotation along the axis of
    cross-product f x t.

    Returns rotation matrix R such that Rf = t

    ----
    Implementation of 'Möller, T., & Hughes, J. F. (1999). Efficiently Building a Matrix to Rotate One Vector to Another. 
    Journal of Graphics Tools, 4(4), 1–4. doi:10.1080/10867651.1999.10487509'
    """
    t = t / np.linalg.norm(t)
    f = f / np.linalg.norm(f)

    # check that the vectors are not already near parallel. 
    if np.allclose(np.abs(np.dot(f, t)), 1, 0.01):
        if verbose == True:
            print("vectors seem to be near parallel, using reflections to compute rotation.")

        # product of two reflections is a rotation matrix, and easier to construct when 
        # vectors are parallel. 
        cond_x = (np.abs(f[0]) <  np.abs(f[1])) and (np.abs(f[0]) < np.abs(f[2]))
        cond_y = (np.abs(f[1]) <  np.abs(f[0])) and (np.abs(f[1]) < np.abs(f[2]))
        cond_z = (np.abs(f[2]) <  np.abs(f[0])) and (np.abs(f[2]) < np.abs(f[1]))

        if cond_x:
            p = np.array([1, 0, 0])
        elif cond_y:
            p = np.array([0, 1, 0])
        elif cond_z:
            p = np.array([0, 0, 1])

        
        # Compute reflections using Householder matrices
        A = Householder(p - f)
        B = Householder(p - t)

        R = B @ A
    
    
    else:
        v = np.cross(f, t)
        c = np.dot(f, t)
        h = (1 - c) / (np.dot(v, v))
    
        R = np.asarray([
        [c+h*v[0]**2, h*v[0]*v[1]-v[2], h*v[0]*v[2]+v[1]],
        [h*v[0]*v[1]+v[2], c+h*v[1]**2, h*v[1]*v[2]-v[0]],
        [h*v[0]*v[2]-v[1], h*v[1]*v[2]+v[0], c+h*v[2]**2]]
        )

    if np.allclose(t, R @ f, 0.001):
        if verbose == "True":
            print("Rotation successful")
        return R
    else:
        print("Rotation unsuccessful")
        return None