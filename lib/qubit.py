import numpy as np
import scipy as sp
from scipy import constants
import qutip as qt
import qutip.settings as settings

settings.atol = 1e-12

class qubit:
    def __init__(self, Ej=153e9, C=[12e-15], ng=0.5, ncut=5):

        self.Ej = Ej * constants.h
        self.C = np.sum(np.array(C))
        self.Ec = constants.e**2 / (2 * self.C)
        self.ng = ng
        self.ncut = ncut

    def print_params(self):

        self.diagonalize_H()
        print(f'Ejp:    {self.Ej * 1e-9 / constants.h} GHz')
        print(f'Ec:    {self.Ec * 1e-9 / constants.h} GHz')
        print(f'Cj:    {self.C * 1e15} fF')
        print(f'Ejp/Ecp : {np.real(self.Ej/self.Ec)}')
        print(f'w_01:    {(self.evals[1] - self.evals[0]) * 1e-9 / constants.h} GHz')
        print(f'ng:    {self.ng}')

    def init_operator(self):

        self.e_iphi_op_cb = qt.qdiags(np.ones(2 * self.ncut), offsets=1)
        self.I_cb = qt.qeye(2 * self.ncut + 1)
        self.n_cb = qt.charge(self.ncut)
        self.ng_cb = self.ng * self.I_cb

    def get_kinetic(self):
        self.init_operator()

        kinetic = 4 * self.Ec * ( self.n_cb - self.ng_cb ) * ( self.n_cb - self.ng_cb )

        return kinetic

    def get_potential(self):
        self.init_operator()

        potential = - 0.5 * self.Ej * (self.e_iphi_op_cb + self.e_iphi_op_cb.trans())

        return potential

    def get_H(self):
        self.Ham = self.get_kinetic() + self.get_potential()

        return self.Ham
    
    def diagonalize_H(self, update=False):
        if update:
            self.Ham = self.get_H()
        else: 
            try:
                self.Ham
            except AttributeError:
                self.Ham = self.get_H()
        
        self.evals, self.evecs = self.Ham.eigenstates()

        return self.evals, self.evecs