import numpy as np
from scipy import constants
import qutip as qt
import qutip as qt
import qutip.settings as settings

from lib.circuit_res_trans import circuit_res_trans

settings.atol = 1e-12 # Setting the absolute tolerance for qutip

class circuit_res: # Class for the circuit-resonator system

    def __init__(self, Cc = [10e-15, 10e-15], Cr=150e-15, Lr=10e-9, mcut=10, qubit_list=[]): # Cc in fF, Cr in fF, Lr in nH, mcut is the number of resonator states to include
        self.Cc1 = Cc[0]
        self.Cc2 = Cc[1]
        self.res_Cr = Cr
        self.res_Lr = Lr
        self.probe = qubit_list[0]
        self.target = qubit_list[1]
        self.mcut = mcut
        self.Z0 = np.sqrt(self.res_Lr / self.res_Cr) # The impedance of the resonator

        self.omega_exp = constants.hbar / np.sqrt(self.res_Lr * self.res_Cr) # The resonator frequency in rad/s

        self.init_operator() # Initialize the charge basis operators for the charge qubit

    def print_params(self): # Print the parameters of the qubit in a nice way
        print("Probe ------------------")
        self.probe.print_params()
        print("Target ------------------")
        self.target.print_params()
        print("Resonator ------------------")
        print(f'wr:    {self.omega_exp * 1e-9 / constants.h} Ghz')
        print(f'Cc1:    {self.Cc1 * 1e15} fF')
        print(f'Cc2:    {self.Cc2 * 1e15} fF')
        print(f'Cr:    {self.res_Cr * 1e15} fF')
        print(f'Lr:    {self.res_Lr * 1e9} nH')
        print(f'Z0:    {self.Z0} Ohm')

    def init_operator(self): # Initialize the charge basis operators for the charge qubit

        self.probe.init_operator() # Initialize the charge basis operators for the charge qubit
        self.target.init_operator() # Initialize the charge basis operators for the charge qubit

        self.I_fb = qt.qeye(self.mcut)
        self.creation_op_fb = qt.create(self.mcut)
        self.annihilation_op_fb = qt.destroy(self.mcut)
        self.q_op_fb = np.sqrt(constants.hbar / (2* self.Z0)) *1j * (self.creation_op_fb - self.annihilation_op_fb)
        self.phi_op_fb = np.sqrt(constants.hbar * self.Z0 * 0.5) * (self.creation_op_fb + self.annihilation_op_fb)

        self.q1_q1_op_qrq = qt.tensor([4 * constants.e **2 * (self.probe.n_cb + self.probe.ng_cb * self.probe.n_cb + self.probe.ng_cb), self.I_fb, self.I_fb, self.target.I_cb])
        self.q1_q2_op_qrq = qt.tensor([2 * constants.e * (self.probe.n_cb + self.probe.ng_cb), self.q_op_fb * self.q_op_fb, self.I_fb, self.target.I_cb])
        self.q1_q3_qrq = qt.tensor([2 * constants.e * (self.probe.n_cb + self.probe.ng_cb), self.q_op_fb, self.q_op_fb, self.target.I_cb])
        self.q1_q4_qrq = qt.tensor([2 * constants.e * (self.probe.n_cb + self.probe.ng_cb), self.q_op_fb, self.I_fb, 2 * constants.e * (self.target.n_cb + self.target.ng_cb)])
        self.q2_q2_qrq = qt.tensor([self.probe.I_cb, self.q_op_fb * self.q_op_fb, self.I_fb, self.target.I_cb])
        self.q2_q3_qrq = qt.tensor([self.probe.I_cb, self.q_op_fb, self.q_op_fb, self.target.I_cb])
        self.q2_q4_qrq = qt.tensor([self.probe.I_cb, self.q_op_fb, self.I_fb, 2 * constants.e * (self.target.n_cb + self.target.ng_cb)])
        self.q3_q3_qrq = qt.tensor([self.probe.I_cb, self.I_fb, self.q_op_fb * self.q_op_fb, self.target.I_cb])
        self.q3_q4_qrq = qt.tensor([self.probe.I_cb, self.I_fb, self.q_op_fb, 2 * constants.e * (self.target.n_cb + self.target.ng_cb)])
        self.q4_q4_qrq = qt.tensor([self.probe.I_cb, self.I_fb, self.I_fb, 4 * constants.e **2 * (self.target.n_cb + self.target.ng_cb) * (self.target.n_cb + self.target.ng_cb)])

        self.q2_q2_r = qt.tensor([self.q_op_fb * self.q_op_fb, self.I_fb])
        self.q2_q3_r = qt.tensor([self.q_op_fb, self.q_op_fb])
        self.q3_q3_r = qt.tensor([self.I_fb, self.q_op_fb * self.q_op_fb])

    def get_kinetic_circuit(self):
        self.init_operator()

        C_mat = [
            [self.probe.C + self.Cc1, -self.Cc1, 0, 0],
            [-self.Cc1, self.Cc1 + self.res_Cr, -self.res_Cr, 0],
            [0, -self.res_Cr, self.res_Cr + self.Cc2, -self.Cc2],
            [0, 0, -self.Cc2, self.Cc2 + self.target.C]
        ] 

        C_mat_inv = np.linalg.inv(C_mat)

        kin = C_mat_inv[0][0] * self.q1_q1_op_qrq
        kin += 2*C_mat_inv[0][1] * self.q1_q2_op_qrq
        kin += 2*C_mat_inv[0][2] * self.q1_q3_qrq
        kin += 2*C_mat_inv[0][3] * self.q1_q4_qrq
        kin += C_mat_inv[1][1] * self.q2_q2_qrq
        kin += 2*C_mat_inv[1][2] * self.q2_q3_qrq
        kin += 2*C_mat_inv[1][3] * self.q2_q4_qrq
        kin += C_mat_inv[2][2] * self.q3_q3_qrq
        kin += 2*C_mat_inv[2][3] * self.q3_q4_qrq
        kin += C_mat_inv[3][3] * self.q4_q4_qrq

        kin *= 0.5
        return kin

    def get_potential_circuit(self):
        self.init_operator()

        pot = -self.probe.Ej * 0.5 * qt.tensor([self.probe.e_iphi_op_cb + self.probe.e_iphi_op_cb.trans(), self.I_fb, self.I_fb, self.target.I_cb])
        pot += -self.target.Ej * 0.5 * qt.tensor([self.probe.I_cb, self.I_fb, self.I_fb, self.target.e_iphi_op_cb + self.target.e_iphi_op_cb.trans()])
        pot += self.probe.Ej * qt.tensor([self.probe.I_cb, self.I_fb, self.I_fb, self.target.I_cb])
        pot += self.target.Ej * qt.tensor([self.probe.I_cb, self.I_fb, self.I_fb, self.target.I_cb])
        pot += 0.5 * self.res_Lr**-1 * qt.tensor([self.probe.I_cb, self.phi_op_fb * self.phi_op_fb, self.I_fb, self.target.I_cb])
        pot += 0.5 * self.res_Lr**-1 * qt.tensor([self.probe.I_cb, self.I_fb, self.phi_op_fb * self.phi_op_fb, self.target.I_cb])
        pot += -self.res_Lr**-1 * qt.tensor([self.probe.I_cb, self.phi_op_fb, self.phi_op_fb, self.target.I_cb])

        return pot

    def get_kinetic_probe(self):
        self.init_operator()

        kin = self.probe.get_kinetic()

        return kin

    def get_potential_probe(self):
        self.init_operator()

        potential = self.probe.get_potential()
        
        return potential

    def get_kinetic_target(self):
        self.init_operator()

        kin = self.target.get_kinetic()
        
        return kin

    def get_potential_target(self):
        self.init_operator()

        potential = self.target.get_potential()
        
        return potential

    def get_kinetic_resonator(self):
        self.init_operator()

        C_mat = [
            [self.Cc1 + self.res_Cr, -self.res_Cr],
            [-self.res_Cr, self.Cc2 + self.res_Cr]
        ]

        C_mat_inv = np.linalg.inv(C_mat)

        kin = C_mat_inv[0][0] * self.q2_q2_r
        kin += 2*C_mat_inv[0][1] * self.q2_q3_r
        kin += C_mat_inv[1][1] * self.q3_q3_r

        kin *= 0.5

        return kin

    def get_potential_resonator(self):
        self.init_operator()

        potential = 0.5 * self.res_Lr**-1 * qt.tensor([self.phi_op_fb * self.phi_op_fb, self.I_fb])
        potential += 0.5 * self.res_Lr**-1 * qt.tensor([self.I_fb, self.phi_op_fb * self.phi_op_fb])
        potential += -self.res_Lr**-1 * qt.tensor([self.phi_op_fb, self.phi_op_fb])
        
        return potential

    ## GET H ##

    def get_H_circuit(self):

        self.H_circuit = self.get_kinetic_circuit() + self.get_potential_circuit()

        return self.H_circuit

    def get_H_probe(self):

        self.H_probe = self.get_kinetic_probe() + self.get_potential_probe()

        return self.H_probe

    def get_H_target(self):

        self.H_target = self.get_kinetic_target() + self.get_potential_target()

        return self.H_target

    def get_H_resonator(self):

        self.H_resonator = self.get_kinetic_resonator() + self.get_potential_resonator()

        return self.H_resonator

    ## DIAGONALISE ##

    def diagonalise_circuit(self,update=False):
        if update:
            self.get_H_circuit()
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        self.eig_values_circuit, self.eig_vectors_circuit = self.H_circuit.eigenstates(eigvals = 10, sparse=True)

        return self.eig_values_circuit, self.eig_vectors_circuit

    def diagonalise_probe(self,update=False):
        if update:
            self.get_H_probe()
        else:
            try:
                self.H_probe
            except AttributeError:
                self.get_H_probe()
        
        self.eig_values_probe, self.eig_vectors_probe = self.H_probe.eigenstates(eigvals = 2, sparse=True)

        return self.eig_values_probe, self.eig_vectors_probe

    def diagonalise_target(self,update=False):
        if update:
            self.get_H_target()
        else:
            try:
                self.H_target
            except AttributeError:
                self.get_H_target()
        
        self.eig_values_target, self.eig_vectors_target = self.H_target.eigenstates(eigvals = self.target.ncut, sparse=True)

        return self.eig_values_target, self.eig_vectors_target

    def diagonalise_resonator(self,update=False):
        if update:
            self.get_H_resonator()
        else:
            try:
                self.H_resonator
            except AttributeError:
                self.get_H_resonator()
        
        self.eig_values_resonator, self.eig_vectors_resonator = self.H_resonator.eigenstates(eigvals = self.mcut, sparse=True)

        return self.eig_values_resonator, self.eig_vectors_resonator

    ## GET QUBIT STATES ##

    def init_qubit_states(self, update=False):
        if update:
            self.diagonalise_probe(update=True)
            self.diagonalise_target(update=True)
            self.diagonalise_circuit(update=True)
            self.diagonalise_resonator(update=True)
        else:
            try:
                self.eig_values_probe
                self.eig_values_target
                self.eig_values_resonator
            except AttributeError:
                self.diagonalise_probe()
                self.diagonalise_target()
                self.diagonalise_circuit()
                self.diagonalise_resonator()

        self.fock_states_res = []
        for k in range(len(self.eig_values_resonator)):
            self.fock_states_res.append(qt.basis(self.mcut, k))

        self.state_product_ebebfb = np.zeros((len(self.eig_vectors_probe), len(self.eig_vectors_resonator), len(self.eig_vectors_target)), dtype=object)

        for i in range(len(self.eig_vectors_probe)):
            for j in range(len(self.eig_vectors_target)):
                for k in range(len(self.eig_vectors_resonator)):
                    self.state_product_ebebfb[i,k,j] = qt.tensor([self.eig_vectors_probe[i],self.fock_states_res[k], self.eig_vectors_target[j]])

    ## CALCULATION OF J matrix element ##
    def get_J(self, update=False):
        if update:
            self.diagonalise_probe(update=True)
            self.diagonalise_target(update=True)
            self.diagonalise_circuit(update=True)
            self.diagonalise_resonator(update=True)
        else:
            try:
                self.eig_values_probe
                self.eig_values_target
                self.eig_values_resonator
            except AttributeError:
                self.diagonalise_probe()
                self.diagonalise_target()
                self.diagonalise_circuit()
                self.diagonalise_resonator()

        self.circ_res_probe = circuit_res_trans(self.Cc1, self.res_Cr, self.res_Lr, self.mcut, self.probe)
        self.circ_res_target = circuit_res_trans(self.Cc2, self.res_Cr, self.res_Lr, self.mcut, self.target)
        self.omega_r = self.circ_res_probe.get_omega_res()
        self.deltas_probe = self.circ_res_probe.get_deltas_transmon()
        self.deltas_target = self.circ_res_target.get_deltas_transmon()
        self.g_probe = self.circ_res_probe.get_g_transmon()
        self.g_target = self.circ_res_target.get_g_transmon()

        self.J = np.zeros((len(self.eig_values_probe), len(self.eig_values_target)), dtype=complex)
        for i in range(len(self.eig_values_probe)):
            for j in range(len(self.eig_values_target)):
                self.J[i,j] = (1 / self.omega_exp) * self.g_probe[i,i] * self.g_target[j,j].conj()
        
        return self.J

    def get_omega_probe(self, update=False): # Get the frequency of the probe qubit
        if update:
            self.diagonalise_probe(update=True)
        else:
            try:
                self.eig_values_probe
            except AttributeError:
                self.diagonalise_probe()
        
        self.omega_probe = self.eig_values_probe[1] - self.eig_values_probe[0]

        return self.omega_probe
    
    def get_omega_target(self, update=False): # Get the frequency of the target qubit
        if update:
            self.diagonalise_target(update=True)
        else:
            try:
                self.eig_values_target
            except AttributeError:
                self.diagonalise_target()

        self.omega_target = self.eig_values_target[1] - self.eig_values_target[0]

        return self.omega_target
    
    def get_detuning_pt(self, update=False): # Get the detuning between the probe and target qubit
        if update:
            self.get_omega_probe(update=True)
            self.get_omega_target(update=True)
        else:
            try:
                self.omega_probe
                self.omega_target
            except AttributeError:
                self.get_omega_probe()
                self.get_omega_target()

        self.detuning_pt = self.omega_probe - self.omega_target

        return self.detuning_pt

    def get_J_forth(self, update=False): # J for the forth order perturbation theory see master thesis for source
        if update:
            self.diagonalise_probe(update=True)
            self.diagonalise_target(update=True)
            self.diagonalise_circuit(update=True)
            self.diagonalise_resonator(update=True)
        else:
            try:
                self.eig_values_probe
                self.eig_values_target
                self.eig_values_resonator
            except AttributeError:
                self.diagonalise_probe()
                self.diagonalise_target()
                self.diagonalise_circuit()
                self.diagonalise_resonator()

        self.circ_res_probe = circuit_res_trans(self.Cc1, self.res_Cr, self.res_Lr, self.mcut, self.probe)
        self.circ_res_target = circuit_res_trans(self.Cc2, self.res_Cr, self.res_Lr, self.mcut, self.target)
        self.omega_r = self.circ_res_probe.get_omega_res()
        self.deltas_probe = self.circ_res_probe.get_deltas_transmon()
        self.deltas_target = self.circ_res_target.get_deltas_transmon()
        self.g_probe = self.circ_res_probe.get_g_transmon()
        self.g_target = self.circ_res_target.get_g_transmon()

        delta1 = self.omega_r - (self.probe.evals[1] - self.probe.evals[0])
        delta2 = self.omega_r - (self.target.evals[1] - self.target.evals[0])



        alpha1 = self.probe.Ec
        alpha2 = self.target.Ec

        g12 = self.g_probe[0,1]
        p12 = self.g_target[0,1]
        # print(delta1, delta2, alpha1, alpha2, g12, p12)
        # print(type(delta1), type(delta2), type(alpha1), type(alpha2), type(g12), type(p12))
        J = 2*g12**2*p12**2*(alpha1*alpha2*(delta1+delta2)**2+alpha1*delta1**2*(delta1+delta2)+alpha2*delta2**2*(delta1+delta2))/(delta1**2*delta2**2*(delta1+delta2)*(delta1-delta2)*(delta2-delta1))

        return J




    # def get_U_circuit(self, update=False):
    #     if update:
    #         self.init_qubit_states(update=True)
    #     else:
    #         try:
    #             self.H_circuit
    #         except AttributeError:
    #             self.get_H_circuit()

    #     self.U_raw = self.H_circuit - qt.tensor(self.H_probe, self.I_fb, self.I_fb, self.I_cb) - qt.tensor(self.I_cb, self.I_fb, self.I_fb, self.H_target) - qt.tensor([self.I_cb, self.H_resonator, self.I_cb])

    #     return self.U_raw

    # def reshape_U(self, update=False):
    #     if update:
    #         self.get_U_circuit(update=True)
    #     else:
    #         try:
    #             self.U_raw
    #         except AttributeError:
    #             self.get_U_circuit()
        
    #     self.U_reshape = np.zeros((24,24))
    #     for i in range(2):
    #         for k in range(3):
    #             for j in range(4):
    #                 self.U_reshapre[i,k,j] = qt.tensor(self.eig_vectors_probe[i], self.fock_states_res[k], self.eig_vectors_target[j]).trans() * self.U_raw * qt.tensor(self.eig_vectors_probe[i], self.fock_states_res[k], self.eig_vectors_target[j])

    #     return self.U_reshape
