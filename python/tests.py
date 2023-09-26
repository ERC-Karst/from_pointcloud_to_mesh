import unittest
import numpy as np

from helpers import pairwise_registration as pr

np.random.seed(10000)

class Test(unittest.TestCase): 
    """
    Basic tests for  pairwise registration
    """
    









    def test_1_register(self):
        print("Start test  1 for pairwise registration")
        # point cloud to be registered
        N = np.random.randint(3,20)
        p1 = np.random.uniform(0,1, (3,N))

        # test 1 : no transformation at all
        R = np.array([[1,0,0],[0,1,0],[0, 0, 1]])
        t = np.zeros((3,1))

        # reference point cloud, known absolutely.
        p2 = R @ p1 + t
        p1_prime, R1,t1 = pr.pairWiseRegistration(p1_t= p1.T, p2_t=p2.T, verbose=False)
        
        print(np.round(R, 3))
        print(np.round(R1, 3))
        self.assertTrue(np.allclose(R,R1, atol=1e-5))
        self.assertTrue(np.allclose(t,t1, atol=1e-5))

    def test_2_register(self):
        print("Start test 2 for pairwise registration")
        # point cloud to be registered
        N = np.random.randint(3,20)
        p1 = np.random.uniform(0,1, (3,N))

    
        # test 2 : no rotation, simple translation 
        R = [[1,0,0],[0,1,0],[0, 0, 1]]
        t = np.random.uniform(-5,5, size=(3,1))
        # reference point cloud, known absolutely.
        p2 = R @ p1 + t
        p1_prime, R1,t1 = pr.pairWiseRegistration(p1_t= p1.T, p2_t=p2.T, verbose=False)

        print(np.round(R, 3))
        print(np.round(R1, 3))
        self.assertTrue(np.allclose(R,R1, atol=1e-5))
        self.assertTrue(np.allclose(t,t1, atol=1e-5))

    def test_3_register(self):
        print("Start test 3 for pairwise registration")
        # point cloud to be registered
        N = np.random.randint(3,20)
        p1 = np.random.uniform(0,1, (3,N))

    
        # test 3 : rotation of pi around x-axis , no translation 
        R = [[1,0,0],[0,0,1],[0, -1, 0]]
        t = np.zeros((3,1))
        # reference point cloud, known absolutely.
        p2 = R @ p1 + t
        p1_prime, R1,t1 = pr.pairWiseRegistration(p1_t= p1.T, p2_t=p2.T, verbose=False)

        print(np.round(R, 3))
        print(np.round(R1, 3))
        self.assertTrue(np.allclose(R,R1, atol=1e-5))
        self.assertTrue(np.allclose(t,t1, atol=1e-5))

    def test_4_register(self):
        print("Start test 4 for pairwise registration")
        # point cloud to be registered
        N = np.random.randint(3,20)
        p1 = np.random.uniform(0,1, (3,N))

    
        # test 4 :  rotation of pi around x-axis and random translation 
        R = [[1,0,0],[0,0,1],[0, -1, 0]]
        t = np.random.uniform(-5,5, size=(3,1))

        # reference point cloud, known absolutely.
        p2 = R @ p1 + t
        p1_prime, R1,t1 = pr.pairWiseRegistration(p1_t= p1.T, p2_t=p2.T, verbose=False)

        print(np.round(R, 3))
        print(np.round(R1, 3))
        self.assertTrue(np.allclose(R,R1, atol=1e-5))
        self.assertTrue(np.allclose(t,t1, atol=1e-5))

    def test_5_register(self):
        print("Start test 5 for pairwise registration")
        # point cloud to be registered
        N = np.random.randint(3,20)
        p1 = np.random.uniform(0,1, (3,N))

            # test 5 : complex rotation from vector[0.39,  0.30, 0.86] to [-0.93 0.28 0.21] and random translation 
        R = np.array([[-0.053172253072, -0.213671043515, -0.975457489491], 
                    [0.579704999924, 0.788778245449, -0.204379230738],
                    [0.813089668751, -0.576344847679, 0.081925034523 ]])
        t = np.random.uniform(-5,5, size=(3,1))

        # reference point cloud, known absolutely.
        p2 = R @ p1 + t
        p1_prime, R1,t1 = pr.pairWiseRegistration(p1_t= p1.T, p2_t=p2.T, verbose=False)
        print(np.round(R, 3))
        print(np.round(R1, 3))
        self.assertTrue(np.allclose(R,R1, atol=1e-5))
        self.assertTrue(np.allclose(t,t1, atol=1e-5))

    def test_6_register(self):
        print("Start test 6 for pairwise registration")
        # point cloud to be registered
        N = np.random.randint(3,20)
        p1 = np.random.uniform(0,1, (3,N))

            # test 6 : complex rotation from vector[0.39,  0.30, 0.86] to [-0.93 0.28 0.21] and random translation 
        R = np.array([[-0.053172253072, -0.213671043515, -0.975457489491], 
                    [0.579704999924, 0.788778245449, -0.204379230738],
                    [0.813089668751, -0.576344847679, 0.081925034523 ]])
        t = np.random.uniform(-5,5, size=(3,1))

        # reference point cloud, known absolutely.
        p2 = R @ p1 + t + np.random.normal(0,0.001, (3, N))
        p1_prime, R1,t1 = pr.pairWiseRegistration(p1_t= p1.T, p2_t=p2.T, verbose=False)
        print(np.round(R, 3))
        print(np.round(R1, 3))
        self.assertTrue(np.allclose(R,R1, atol=1e-2))
        self.assertTrue(np.allclose(t,t1, atol=1e-2))


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()