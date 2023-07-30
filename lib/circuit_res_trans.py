import numpy as np
import scipy as sp
from scipy import constants
import qutip as qt
import qutip as qt
import qutip.settings as settings
from qutip.ui.progressbar import TextProgressBar as ProgressBar
settings.atol = 1e-12

class circuit_res_trans:

    ## INITIALIZATION ##

    def __init__(self, Cc, Cr, Lr, mcut, qubit):

        self.qubit = qubit[0]
        self.Cc = Cc
        self.res_Cr = Cr
        self.res_Lr = Lr
        self.mcut = mcut
        self.Z0 = np.sqrt(self.res_Lr / self.res_Cr)
        
        self.init_operator()

    def print_params(self):
            
        self.qubit.print_params()
        print(f'Cc:    {self.Cc * 1e15} fF')
        print(f'Cr:    {self.res_Cr * 1e15} fF')
        print(f'Lr:    {self.res_Lr * 1e9} nH')
        print(f'Z0:    {self.Z0} Ohm')

    def init_operator(self):

        self.qubit.init_operator()

        self.I_cb = self.qubit.I_cb
        self.I_fb = qt.qeye(self.mcut)
        self.res_create = qt.create(self.mcut)
        self.q_op_fb = np.sqrt(constants.hbar / (2 * self.Z0)) * 1j * (self.res_create - self.res_create.dag())
        self.phi_op_fb = np.sqrt(constants.hbar * self.Z0 / 2) * (self.res_create + self.res_create.dag())

        self.q1_q1 = qt.tensor([4 * constants.e **2 *(self.qubit.n_cb + self.qubit.ng_cb) * (self.qubit.n_cb + self.qubit.ng_cb), self.I_fb])
        self.q1_q2 = qt.tensor([2 * constants.e * (self.qubit.n_cb + self.qubit.ng_cb), self.q_op_fb]) 
        self.q2_q2 = qt.tensor([self.I_cb, self.q_op_fb * self.q_op_fb])

        ## GET KIN AND POT ##

    def get_kinetic(self):
        self.init_operator()

        C_mat = [
            [self.qubit.C + self.Cc, - self.Cc],
            [-self.Cc, self.res_Cr + self.Cc] 
        ]

        C_mat_inverse = np.linalg.inv(C_mat)

        kin = C_mat_inverse[0][0] * self.q1_q1
        kin += 2 * C_mat_inverse[0][1] * self.q1_q2
        kin += C_mat_inverse[1][1] * self.q2_q2

        kin *= 0.5 

        return kin
    
    def get_potential(self):
        self.init_operator()

        pot = -0.5 * self.qubit.Ej * qt.tensor(self.qubit.e_iphi_op_cb + self.qubit.e_iphi_op_cb.trans(), self.I_fb)
        pot += 0.5 * self.res_Lr**-1 * qt.tensor([self.I_cb, self.phi_op_fb * self.phi_op_fb])

        return pot
    
    def get_kinetic_qubit(self):
        self.init_operator()

        kin = self.qubit.get_kinetic()

        return kin

    def get_potential_qubit(self):
        self.init_operator()

        potential = self.qubit.get_potential()
        
        return potential

    def get_kinetic_res(self):
        self.init_operator()

        kin = self.q_op_fb**2 / (2 * self.res_Cr)

        return kin

    def get_potential_res(self):
        self.init_operator()

        pot = self.phi_op_fb**2 / ( 2 * self.res_Lr)

        return pot
    
    ## GET H ##

    def get_H_circuit(self):
        self.init_operator()

        self.H_circuit = self.get_kinetic() + self.get_potential()

        return self.H_circuit
    
    def get_H_qubit(self):

        self.H_qubit = self.qubit.get_H()

        return self.H_qubit
    
    def get_H_res(self):

        self.H_res = self.get_kinetic_res() + self.get_potential_res()

        return self.H_res
    
    ## DIAGONALISE ## 

    def diagonalise_circuit(self, update=False):
        if update:
            self.get_H_circuit()
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        self.eig_values_circuit, self.eig_vectors_circuit = self.H_circuit.eigenstates(eigvals = 10)

        return self.eig_values_circuit, self.eig_vectors_circuit

    def diagonalise_qubit(self,update=False):
        if update:
            self.get_H_qubit()
        else:
            try:
                self.H_qubit
            except AttributeError:
                self.get_H_qubit()

        self.eig_values_qubit, self.eig_vectors_qubit = self.H_qubit.eigenstates(eigvals = 10)

        return self.eig_values_qubit, self.eig_vectors_qubit
    
    def diagonalise_res(self,update=False):
        if update:
            self.get_H_res()
        else:
            try:
                self.H_res
            except AttributeError:
                self.get_H_res()

        self.eig_values_res, self.eig_vectors_res = self.H_res.eigenstates(eigvals = 10)

        return self.eig_values_res, self.eig_vectors_res
    
    ## GET QUBIT STATES ## 

    def get_states(self, update=False):
        if update:
            self.diagonalise_qubit(update)
            self.diagonalise_circuit(update)
            self.diagonalise_res(update)
        else:
            try:
                self.eig_values_qubit
                self.eig_values_circuit
                self.eig_values_res
            except AttributeError:
                self.diagonalise_qubit()
                self.diagonalise_circuit()
                self.diagonalise_res()

        self.states_00_ebfb = qt.tensor(self.eig_vectors_qubit[0],qt.basis(self.mcut,0))
        self.states_01_ebfb = qt.tensor(self.eig_vectors_qubit[0],qt.basis(self.mcut,1))
        self.states_11_ebfb = qt.tensor(self.eig_vectors_qubit[1],qt.basis(self.mcut,1))
        self.states_qubit_p = 2 ** -0.5 * (self.eig_vectors_qubit[0] + self.eig_vectors_qubit[1])
        self.states_qubit_m = 2 ** -0.5 * (self.eig_vectors_qubit[0] - self.eig_vectors_qubit[1])
        self.states_p1_ebfb =  qt.tensor(self.states_qubit_p,qt.basis(self.mcut,1))
        self.states_m0_ebfb =  qt.tensor(self.states_qubit_m,qt.basis(self.mcut,0))

    ## QUBIT CAVITY COUPLIGN CALCULUS ##

    def get_delta_qubit(self, update=False):
        if update:
            self.get_states(update)
        else:
            try:
                self.states_00_ebfb
            except AttributeError:
                self.get_states()
        
        self.delta_qubit = 2 * ( self.states_00_ebfb.dag() * self.H_circuit * self.states_00_ebfb )
        self.delta_qubit = self.delta_qubit.full()[0,0]

        return self.delta_qubit
    
    def get_omega_res(self, update=False):
        if update:
            self.get_states(update)
        else:
            try:
                self.states_01_ebfb
            except AttributeError:
                self.get_states()
        
        self.omega_res = ( self.states_01_ebfb.dag() * self.H_circuit * self.states_01_ebfb )
        self.omega_res -= ( self.states_00_ebfb.dag() * self.H_circuit * self.states_00_ebfb )
        self.omega_res = self.omega_res.full()[0,0]

        return self.omega_res
    
    def get_gparr(self, update=False):
        if update:
            self.get_states(update)
        else:
            try:
                self.states_p1_ebfb
            except AttributeError:
                self.get_states()
        
        self.gparr = ( self.states_p1_ebfb.dag() * self.H_circuit * self.states_m0_ebfb )
        self.gparr = self.gparr.full()[0,0]

        return self.gparr
    
    def get_gperp(self, update=False):
        if update:
            self.get_states(update)
        else:
            try:
                self.states_11_ebfb
            except AttributeError:
                self.get_states()
        
        self.gperp = ( self.states_11_ebfb.dag() * self.H_circuit * self.states_00_ebfb )
        self.gperp = self.gperp.full()[0,0]

        return self.gperp

    ## TRANSMON RAW CAVITY COUPLING TERM CALCULUS ##


    # Not very usefull, a function to check the shifht with the circuit should not change that much
    def get_deltas_transmon(self, update=False):
        if update:
            self.diagonalise_qubit(update)
            self.get_H_circuit()
        else:
            try:
                self.eig_values_qubit
            except AttributeError:
                self.diagonalise_qubit()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        self.deltas_transmon = np.zeros(len(self.eig_values_qubit), dtype=complex)
        for i in range(len(self.eig_values_qubit)):
            self.deltas_transmon[i] = (qt.tensor(self.eig_vectors_qubit[i], qt.basis(self.mcut, 0)).dag() * self.H_circuit * qt.tensor(self.eig_vectors_qubit[i], qt.basis(self.mcut, 0))).full()[0,0]

        return self.deltas_transmon
    
    def get_omega_res_secondcalc(self, update=False):
        if update:
            self.diagonalise_qubit(update)
            self.get_H_circuit(update)
        else:
            try:
                self.eig_values_qubit
            except AttributeError:
                self.diagonalise_qubit()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()


        self.omega_res_secondcalc = qt.tensor(self.eig_vectors_qubit[0], qt.basis(self.mcut, 1)).dag() * self.H_circuit * qt.tensor(self.eig_vectors_qubit[0], qt.basis(self.mcut, 1))
        self.omega_res_secondcalc -= qt.tensor(self.eig_vectors_qubit[0], qt.basis(self.mcut, 0)).dag() * self.H_circuit * qt.tensor(self.eig_vectors_qubit[0], qt.basis(self.mcut, 0))
        self.omega_res_secondcalc = float(np.absolute(self.omega_res_secondcalc.full()))
        return self.omega_res_secondcalc


    def get_g_transmon(self, update=False):
        if update:
            self.diagonalise_qubit(update)
            self.get_H_circuit(update)
        else:
            try:
                self.eig_values_qubit
            except AttributeError:
                self.diagonalise_qubit()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        m = len(self.eig_values_qubit)
        self.g_transmon = np.zeros((m,m), dtype=complex)
        for i in range(m):
            for j in range(m):
                self.g_transmon[i,j] = (qt.tensor(self.eig_vectors_qubit[i], qt.basis(self.mcut, 1)).dag() * self.H_circuit * qt.tensor(self.eig_vectors_qubit[j], qt.basis(self.mcut, 0))).full()[0,0]

        return self.g_transmon