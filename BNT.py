import numpy as np

class BNT():

    def __init__(self, z, chi, n_i_list):

        self.z = z
        self.chi = chi
        self.n_i_list = n_i_list
        self.nbins = len(n_i_list)


    def get_matrix(self):

        #analagous to 1, 1/chi_i in the discrete limit
        #equation 7 arxiv 1312.0430
        A_list = []
        B_list = []
        for i in range(self.nbins):
            nz = self.n_i_list[i]
            A_list += [np.trapz(nz, self.z)]
            B_list += [np.trapz(nz / self.chi, self.z)]

        # set the initial conditions for the matrix
        # equation 12 arxiv 1312.0430
        BNT_matrix = np.eye(self.nbins)
        BNT_matrix[1,0] = -1.

        #solve the resulting system
        #in the countinuos limit analagous to eqs 13, 14, 15
        for i in range(2,self.nbins):
            mat = np.array([ [A_list[i-1], A_list[i-2]], [B_list[i-1], B_list[i-2]] ])
            A = -1. * np.array( [A_list[i], B_list[i]] )
            soln = np.dot(np.linalg.inv(mat), A)
            BNT_matrix[i,i-1] = soln[0]
            BNT_matrix[i,i-2] = soln[1]
        
        return BNT_matrix


