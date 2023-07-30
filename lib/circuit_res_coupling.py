import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy import constants
import qutip as qt
import qutip as qt
import qutip.settings as settings
from qutip.ui.progressbar import TextProgressBar as ProgressBar
settings.atol = 1e-12

class circuit_res:

    def __init__(self, Cc = [10e-15,10e-15], ng=0.5, Cr=150e-15, Lr=10e-9, mcut=10, transmon_list=[]):
        self.Cc1 = Cc[0]
        self.Cc2 = Cc[1]
        self.ng = ng
        self.res_Cr = Cr
        self.res_Lr = Lr
        self.probe = transmon_list[0]
        self.target = transmon_list[1]
        self.ncut = self.probe.ncut
        self.mcut = mcut
        self.Z0 = np.sqrt(self.res_Lr / self.res_Cr)

        self.init_operator()

    def init_operator(self):

        self.I_cb = qt.qeye(2 * self.ncut + 1)
        self.I_fb = qt.qeye(self.mcut)

        self.q_op_cb = 2 * qt.charge(self.ncut)
        self.qg_op_cb = 2 *self.ng * self.I_cb
        self.e_iphi_op_cb =  qt.qdiags(np.ones(2*self.ncut), offsets=1)
        self.creation_op_fb = qt.create(self.mcut)
        self.annihilation_op_fb = qt.destroy(self.mcut)
        self.q_op_fb = np.sqrt(constants.hbar / (2* self.Z0)) *1j * (self.creation_op_fb - self.annihilation_op_fb)
        self.phi_op_fb = np.sqrt(constants.hbar * self.Z0 * 0.5) * (self.creation_op_fb + self.annihilation_op_fb)

        self.q1_q1_op_qrq = qt.tensor([self.q_op_cb * self.q_op_cb, self.I_fb, self.I_fb, self.I_cb])
        self.q1_q2_op_qrq = qt.tensor([self.q_op_cb, self.q_op_fb * self.q_op_fb, self.I_fb, self.I_cb])
        self.q1_q3_qrq = qt.tensor([self.q_op_cb, self.q_op_fb, self.q_op_fb, self.I_cb])
        self.q1_q4_qrq = qt.tensor([self.q_op_cb, self.q_op_fb, self.I_fb, self.q_op_cb])
        self.q2_q2_qrq = qt.tensor([self.I_cb, self.q_op_fb * self.q_op_fb, self.I_fb, self.I_cb])
        self.q2_q3_qrq = qt.tensor([self.I_cb, self.q_op_fb, self.q_op_fb, self.I_cb])
        self.q2_q4_qrq = qt.tensor([self.I_cb, self.q_op_fb, self.I_fb, self.q_op_cb])
        self.q3_q3_qrq = qt.tensor([self.I_cb, self.I_fb, self.q_op_fb * self.q_op_fb, self.I_cb])
        self.q3_q4_qrq = qt.tensor([self.I_cb, self.I_fb, self.q_op_fb, self.q_op_cb])
        self.q4_q4_qrq = qt.tensor([self.I_cb, self.I_fb, self.I_fb, self.q_op_cb * self.q_op_cb])

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

        kin *= 0.5*constants.e*constants.e
        return kin

    def get_potential_circuit(self):
        self.init_operator()

        pot = -self.probe.Ej * 0.5 * qt.tensor([self.e_iphi_op_cb + self.e_iphi_op_cb.trans(), self.I_fb, self.I_fb, self.I_cb])
        pot += -self.target.Ej * 0.5 * qt.tensor([self.I_cb, self.I_fb, self.I_fb, self.e_iphi_op_cb + self.e_iphi_op_cb.trans()])
        pot += self.probe.Ej * qt.tensor([self.I_cb, self.I_fb, self.I_fb, self.I_cb])
        pot += self.target.Ej * qt.tensor([self.I_cb, self.I_fb, self.I_fb, self.I_cb])
        pot += 0.5 * self.res_Lr**-1 * qt.tensor([self.I_cb, self.phi_op_fb * self.phi_op_fb, self.I_fb, self.I_cb])
        pot += 0.5 * self.res_Lr**-1 * qt.tensor([self.I_cb, self.I_fb, self.phi_op_fb * self.phi_op_fb, self.I_cb])
        pot += -self.res_Lr**-1 * qt.tensor([self.I_cb, self.phi_op_fb, self.phi_op_fb, self.I_cb])

        return pot

    def get_kinetic_probe(self):
        self.init_operator()
        settings.atol = 1e-12  # Adjust this value to your desired tolerance level
        kin = 0.5 * constants.e *constants.e *((self.q_op_cb  ) * (self.q_op_cb  )) / (self.probe.C + self.Cc1)
        return kin

    def get_potential_probe(self):
        self.init_operator()

        potential = self.probe.Ej * self.I_cb
        potential += -self.probe.Ej * (self.e_iphi_op_cb + self.e_iphi_op_cb.trans())
        
        return potential

    def get_kinetic_target(self):
        self.init_operator()

        kin = 0.5 *constants.e * constants.e *(self.q_op_cb * self.q_op_cb) / (self.target.C + self.Cc2)
        
        return kin

    def get_potential_target(self):
        self.init_operator()

        potential = self.target.Ej * self.I_cb
        potential += -self.target.Ej * (self.e_iphi_op_cb + self.e_iphi_op_cb.trans())
        
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

        kin *= 0.5*constants.e **2

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
        
        self.eig_values_target, self.eig_vectors_target = self.H_target.eigenstates(eigvals = 4, sparse=True)

        return self.eig_values_target, self.eig_vectors_target

    def diagonalise_resonator(self,update=False):
        if update:
            self.get_H_resonator()
        else:
            try:
                self.H_resonator
            except AttributeError:
                self.get_H_resonator()
        
        self.eig_values_resonator, self.eig_vectors_resonator = self.H_resonator.eigenstates(eigvals = 3, sparse=True)

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

    ## CALCULATION ##

    def get_U_circuit(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        self.U_raw = self.H_circuit - qt.tensor(self.H_probe, self.I_fb, self.I_fb, self.I_cb) - qt.tensor(self.I_cb, self.I_fb, self.I_fb, self.H_target) - qt.tensor([self.I_cb, self.H_resonator, self.I_cb])

        return self.U_raw

    def reshape_U(self, update=False):
        if update:
            self.get_U_circuit(update=True)
        else:
            try:
                self.U_raw
            except AttributeError:
                self.get_U_circuit()
        
        self.U_reshape = np.zeros((24,24))
        for i in range(2):
            for k in range(3):
                for j in range(4):
                    self.U_reshapre[i,k,j] = qt.tensor(self.eig_vectors_probe[i], self.fock_states_res[k], self.eig_vectors_target[j]).trans() * self.U_raw * qt.tensor(self.eig_vectors_probe[i], self.fock_states_res[k], self.eig_vectors_target[j])

        return self.U_reshape
