import numpy as np
import scipy as sp
import scipy.sparse as sparse
from scipy import constants
import qutip as qt
import qutip as qt
import qutip.settings as settings
from qutip.ui.progressbar import TextProgressBar as ProgressBar
settings.atol = 1e-12

class circuit_cap:

    ## INITIALIZATION ##

    def __init__(self, Cc=3.1e-15, ng=0.5, transmon_list=[]):
        self.transmon_list = transmon_list
        self.Cc = Cc
        self.ng = ng
        self.probe = transmon_list[0]
        self.target = transmon_list[1]
        self.ncut = self.probe.ncut

        self.init_operator()

    def print_params(self):
        self.get_detunning(update=True)
        print(f'Ejp:    {self.probe.Ej * 1e-9 / constants.h} GHz')
        print(f'Ec:    {constants.e**2/(2*(self.probe.C+self.Cc)) * 1e-9 / constants.h} GHz')
        print(f'Cjp:    {self.probe.C * 1e15} fF')
        print(f'Ejp/Ecp probe: {np.real(self.probe.Ej/(constants.e**2/(2*(self.probe.C+self.Cc))))}')
        print(f'w_probe:    {self.omega_probe * 1e-9 / constants.h} GHz')

        print(f'Ejt:    {self.target.Ej * 1e-9 / constants.h} GHz')
        print(f'Ec:    {constants.e**2/(2*(self.target.C+self.Cc)) * 1e-9 / constants.h} GHz')
        print(f'Cjt:    {self.target.C * 1e15} fF')
        print(f'Ejt/Ect target: {np.real(self.target.Ej/(constants.e**2/(2*(self.target.C+self.Cc))))}')
        print(f'w_target:    {self.omega_target * 1e-9 / constants.h} GHz')

        print(f'detunning:    {self.detunning * 1e-9 / constants.h} GHz')

        print(f'Cc:    {self.Cc * 1e15} fF')
        print(f'ng:    {self.ng}')




    def init_operator(self):
        self.I_cb = sp.sparse.diags(np.ones(2 * self.ncut + 1))   # Identity for qubit (charge basis)

        self.q_op_cb = sp.sparse.diags(2 *  np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_))           # Charge operator (charge basis)
        self.n_op_cb = sp.sparse.diags(2 * np.arange(-self.ncut, self.ncut + 1, dtype=np.complex_))           # Charge operator (charge basis)
        self.ng_op_cb = 2 *  self.ng * self.I_cb
        self.e_iphi_op_cb = sp.sparse.diags(np.ones(2 * self.ncut, dtype=np.complex_), offsets=1)

        self.q1_q1 = sparse.kron((self.q_op_cb )@ (self.q_op_cb ), self.I_cb)
        self.q1_q2 = sparse.kron(self.q_op_cb , self.q_op_cb)
        self.q2_q2 = sparse.kron(self.I_cb, self.q_op_cb @ self.q_op_cb)

    def hc(self, state):
        return np.conjugate(state).T

    def mod_squared(self, val):
        return np.real(val * np.conjugate(val))

    ## GET KIN AND POT ##

    def get_kinetic_circuit(self):
        self.init_operator()

        C_mat = [
            [self.probe.C, + self.Cc -self.Cc],
            [-self.Cc, self.target.C + self.Cc]
        ]

        C_mat_inverse = np.linalg.inv(C_mat)
        
        kin = C_mat_inverse[0][0] * self.q1_q1
        kin += 2 * C_mat_inverse[0][1] * self.q1_q2
        kin += C_mat_inverse[1][1] * self.q2_q2

        kin *= constants.e *0.5 *constants.e 

        return kin

    def get_potential_circuit(self):
        self.init_operator()

        potential = self.probe.Ej * sparse.kron(self.I_cb, self.I_cb)
        potential += self.target.Ej * sparse.kron(self.I_cb, self.I_cb)
        potential += -self.probe.Ej * sparse.kron(self.e_iphi_op_cb + self.e_iphi_op_cb.T, self.I_cb)
        potential += -self.target.Ej * sparse.kron(self.I_cb, self.e_iphi_op_cb + self.e_iphi_op_cb.T)

        return potential

    def get_kinetic_probe(self):
        self.init_operator()

        kin = 0.5 * constants.e *constants.e *((self.q_op_cb  ) @ (self.q_op_cb  )) / (self.probe.C + self.Cc)

        return kin

    def get_potential_probe(self):
        self.init_operator()

        potential = self.probe.Ej * self.I_cb
        potential += -self.probe.Ej * (self.e_iphi_op_cb + self.e_iphi_op_cb.T)
        
        return potential

    def get_kinetic_target(self):
        self.init_operator()

        kin = 0.5 *constants.e * constants.e *(self.q_op_cb @ self.q_op_cb) / (self.target.C + self.Cc)
        
        return kin

    def get_potential_target(self):
        self.init_operator()

        potential = self.target.Ej * self.I_cb
        potential += -self.target.Ej * (self.e_iphi_op_cb + self.e_iphi_op_cb.T)
        
        return potential

    ## GET H ##

    def get_H_circuit(self):
        self.H_circuit = self.get_kinetic_circuit() + self.get_potential_circuit()
        self.H_circuit.eliminate_zeros()
        
        return self.H_circuit
    
    def get_H_probe(self):
        self.H_probe = self.get_kinetic_probe() + self.get_potential_probe()
        self.H_probe.eliminate_zeros()
        
        return self.H_probe
    
    def get_H_target(self):
        self.H_target = self.get_kinetic_target() + self.get_potential_target()
        self.H_target.eliminate_zeros()
        
        return self.H_target

    ## DIAGONALISE ##

    def diagonalise_circuit(self,update=False):
        if update:
            self.get_H_circuit()
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        self.eig_values_circuit, eig_vectors_circuit = sp.sparse.linalg.eigsh(self.H_circuit, k=5, which='SR')
        eig_vectors_circuit = eig_vectors_circuit.T
        self.eig_vectors_circuit = [sp.sparse.csr_array(eig_vectors_circuit[0]), sp.sparse.csr_array(eig_vectors_circuit[1])]

        return self.eig_values_circuit, self.eig_vectors_circuit

    def diagonalise_probe(self,update=False):
        if update:
            self.get_H_probe()
        else:
            try:
                self.Ham_probe
            except AttributeError:
                self.get_H_probe()
        
        self.eig_values_probe, eig_vectors_probe = sp.sparse.linalg.eigsh(self.H_probe, k=2, which='SR') # We are only interested in the two lowest energy levels
        eig_vectors_probe = eig_vectors_probe.T
        self.eig_vectors_probe = eig_vectors_probe

        return self.eig_values_probe, self.eig_vectors_probe

    def diagonalise_target(self,update=False):
        if update:
            self.get_H_target()
        else:
            try:
                self.H_target
            except AttributeError:
                self.get_H_target()
        
        self.eig_values_target, eig_vectors_target = sp.sparse.linalg.eigsh(self.H_target, k=15, which='SR')
        eig_vectors_target = eig_vectors_target.T
        self.eig_vectors_target = eig_vectors_target

        return self.eig_values_target, self.eig_vectors_target

    ## GET QUBIT STATES ##

    def init_qubit_states(self, update=False):
        if update:
            self.diagonalise_probe(update=True)
            self.diagonalise_target(update=True)
            self.diagonalise_circuit(update=True)
        else:
            try:
                self.eig_values_probe
                self.eig_values_target
            except AttributeError:
                self.diagonalise_probe()
                self.diagonalise_target()
                self.diagonalise_circuit()

        self.state_p0eb = self.eig_vectors_probe[0].T
        self.state_p1eb = self.eig_vectors_probe[1].T
        self.state_target = []

        for i in range(len(self.eig_vectors_target)):
            self.state_target.append(self.eig_vectors_target[i].T)

        self.state_product_ebeb = []

        for i in range(len(self.eig_vectors_probe)):
            for j in range(len(self.eig_vectors_target)):
                self.state_product_ebeb.append(sp.sparse.kron(self.eig_vectors_probe[i], self.eig_vectors_target[j]).T)

        self.state_target_0_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        self.state_target_0_cb[self.ncut] = 1
        self.state_target_1_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        self.state_target_1_cb[self.ncut + 1] = 1

        c = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        c[self.ncut] = 1
        self.state_product_ebcb = []

        for i in range(len(self.eig_vectors_probe)):
            for j in range(2 * self.ncut + 1):
                self.state_product_ebcb.append(sp.sparse.kron(self.eig_vectors_probe[i], c.T).T)

    ## CALCULATION ##

    def get_U_circuit(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_p0eb
                self.state_p1eb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        self.U_raw = self.H_circuit - sp.sparse.kron(self.H_probe, self.I_cb) - sp.sparse.kron(self.I_cb, self.H_target)

        size_ebcb = len(self.state_product_ebcb)
        self.U_ebcb = np.zeros((size_ebcb, size_ebcb), dtype=object)

        for i in range(size_ebcb):
            for j in range(size_ebcb):
                self.U_ebcb[i,j] = np.real(self.hc(self.state_product_ebcb[i]).dot(self.U_raw.dot(self.state_product_ebcb[j])).toarray()[0][0])
        
        size_ebeb = len(self.state_product_ebeb)
        self.U_ebeb = np.zeros((size_ebeb, size_ebeb), dtype=object)

        for i in range(size_ebeb):
            for j in range(size_ebeb):
                self.U_ebeb[i,j] = np.real(self.hc(self.state_product_ebeb[i]).dot(self.U_raw.dot(self.state_product_ebeb[j])).toarray()[0][0])

        return self.U_ebcb, self.U_ebeb, self.U_raw

    def extract_U_zz(self, update=False):
        if update:
            self.get_U_circuit(update=True)
        else:
            try:
                self.U_ebcb
                self.U_ebeb
            except AttributeError:
                self.get_U_circuit()
        
        size_ebeb = len(self.state_product_ebeb)
        self.U_zz_ebeb = np.zeros((size_ebeb, size_ebeb), dtype=object)

        for i in range(size_ebeb):
            self.U_zz_ebeb[i,i] = self.U_ebeb[i,i]

        size_ebcb = len(self.state_product_ebcb)
        self.U_zz_ebcb = np.zeros((size_ebcb, size_ebcb), dtype=object)

        for i in range(size_ebcb):
            self.U_zz_ebcb[i,i] = self.U_ebcb[i,i]

        return self.U_zz_ebcb, self.U_zz_ebeb

    def extract_U_zx(self, update=False):
        if update:
            self.get_U_circuit(update=True)
        else:
            try:
                self.U_ebcb
                self.U_ebeb
            except AttributeError:
                self.get_U_circuit()
        
        size_ebeb = len(self.state_product_ebeb)
        self.U_zx_ebeb = np.zeros((size_ebeb, size_ebeb), dtype=object)

        for i in range(size_ebeb):
            for j in range(size_ebeb):
                if i != j:
                    self.U_zx_ebeb[i,j] = self.U_ebeb[i,j]

        size_ebcb = len(self.state_product_ebcb)
        self.U_zx_ebcb = np.zeros((size_ebcb, size_ebcb), dtype=object)
        
        for i in range(size_ebcb):
            for j in range(size_ebcb):
                if i != j:
                    self.U_zx_ebcb[i,j] = self.U_ebcb[i,j]

        return self.U_zx_ebcb, self.U_zx_ebeb

    def get_g_parr(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_p0eb
                self.state_p1eb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()
        U_cache = self.H_circuit - sp.sparse.kron(self.H_probe, self.I_cb) - sp.sparse.kron(self.I_cb, self.H_target)
        self.g_parr = np.abs(np.real(self.hc(sp.sparse.kron(self.eig_vectors_probe[0], self.eig_vectors_target[0]).T).dot(U_cache.dot(sp.sparse.kron(self.eig_vectors_probe[0], self.eig_vectors_target[0]).T)).toarray()[0][0]))

        return self.g_parr

    def get_g_perp(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_p0eb
                self.state_p1eb
            except AttributeError:
                self.init_qubit_states()

        U_cache = self.H_circuit - sp.sparse.kron(self.H_probe, self.I_cb) - sp.sparse.kron(self.I_cb, self.H_target)
        self.g_perp = np.abs(np.real(self.hc(sp.sparse.kron(self.eig_vectors_probe[0], self.eig_vectors_target[0]).T).dot(U_cache.dot(sp.sparse.kron(self.eig_vectors_probe[1], self.eig_vectors_target[0]).T)).toarray()[0][0]))

        return self.g_perp
    
    def get_omega_probe(self, update=False):
        if update:
            self.diagonalise_probe(update=True)
        else:
            try:
                self.eig_values_probe
            except AttributeError:
                self.diagonalise_probe()
        self.omega_probe = self.eig_values_probe[1] - self.eig_values_probe[0]

        return self.omega_probe
    
    def get_omega_target(self, update=False):
        if update:
            self.diagonalise_target(update=True)
        else:
            try:
                self.eig_values_target
            except AttributeError:
                self.diagonalise_target()
        self.omega_target = self.eig_values_target[1] - self.eig_values_target[0]
        return self.omega_target
    
    def get_detunning(self, update=False):
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

        self.detunning = self.omega_probe - self.omega_target
        return self.detunning
    
    def test(self):
        self.get_H_circuit()
        self.diagonalise_probe()
        self.diagonalise_target()

        self.probe_0_eb = self.eig_vectors_probe[0]
        self.probe_1_eb = self.eig_vectors_probe[1]
        print(self.probe_0_eb)
        print(self.probe_1_eb)
        self.probe_p_eb = 2**-0.5 * (self.probe_0_eb + self.probe_1_eb)
        self.probe_m_eb = 2**-0.5 * (self.probe_0_eb - self.probe_1_eb)
        print(self.probe_p_eb)
        print(self.probe_m_eb)

        self.target_0_cb = sp.sparse.csr_matrix((2 * self.ncut + 1, 1))
        self.target_0_cb[self.ncut, 0] = 1
        self.target_1_cb = sp.sparse.csr_matrix((2 * self.ncut + 1, 1))
        self.target_1_cb[self.ncut + 1, 0] = 1
        print(self.target_0_cb)
        print(self.target_1_cb)

        self.target_0_eb = self.eig_vectors_target[0]
        self.target_1_eb = self.eig_vectors_target[1]
        print(self.probe_0_eb)
        print(self.probe_1_eb)
        self.target_p_eb = 2**-0.5 * (self.target_0_eb + self.target_1_eb)
        self.target_m_eb = 2**-0.5 * (self.target_0_eb - self.target_1_eb)
        self.eb_cb_p0 = sp.sparse.kron(self.probe_p_eb, self.target_0_cb.T).T
        self.eb_cb_p1 = sp.sparse.kron(self.probe_p_eb, self.target_1_cb.T).T
        self.eb_cb_m0 = sp.sparse.kron(self.probe_m_eb, self.target_0_cb.T).T
        self.eb_cb_m1 = sp.sparse.kron(self.probe_m_eb, self.target_1_cb.T).T
        self.eb_eb_pp = sp.sparse.kron(self.probe_p_eb, self.target_p_eb.T).T
        self.eb_eb_pm = sp.sparse.kron(self.probe_p_eb, self.target_m_eb.T).T
        self.eb_eb_mp = sp.sparse.kron(self.probe_m_eb, self.target_p_eb.T).T
        self.eb_eb_mm = sp.sparse.kron(self.probe_m_eb, self.target_m_eb.T).T
        print(self.eb_cb_p0.shape)
        print(self.eb_cb_p1.shape)
        print(self.eb_cb_m0.shape)
        print(self.eb_cb_m1.shape)
        print(self.H_circuit.shape)
        print(self.H_circuit.dot(self.eb_cb_p1).shape)
        g_parr = self.hc(self.eb_cb_p1).dot(self.H_circuit.dot(self.eb_cb_m1)).toarray()[0][0] - self.hc(self.eb_cb_p0).dot(self.H_circuit.dot(self.eb_cb_m0)).toarray()[0][0]
        g_parr2 = self.hc(self.eb_eb_pp).dot(self.H_circuit.dot(self.eb_eb_mm)).toarray()[0][0]
        return g_parr, g_parr2
    