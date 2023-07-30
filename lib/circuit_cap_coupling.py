import numpy as np
import scipy as sp
from scipy import constants
import qutip as qt
import qutip as qt
import qutip.settings as settings
settings.atol = 1e-12

class circuit_cap:

    ## INITIALIZATION ##

    def __init__(self, Cc=3.1e-15, qubit_list=[]):

        self.qubit_list = qubit_list
        self.Cc = Cc
        self.probe = self.qubit_list[0]
        self.target = self.qubit_list[1]
        self.ncut = self.probe.ncut

        self.init_operator()

    def print_params(self):

        self.get_detunning(update=True)
        print(f'Ejp:    {self.probe.Ej * 1e-9 / constants.h} GHz')
        print(f'Ecp:    {self.probe.Ec * 1e-9 / constants.h} GHz')
        print(f'Cjp:    {self.probe.C * 1e15} fF')
        print(f'Ejp/Ecp probe: {np.real(self.probe.Ej/self.probe.Ec)}')
        print(f'w_probe:    {self.omega_probe * 1e-9 / constants.h} GHz')
        print(f'ng probe:    {self.probe.ng}')

        print(f'Ejt:    {self.target.Ej * 1e-9 / constants.h} GHz')
        print(f'Ec:    {self.target.Ec * 1e-9 / constants.h} GHz')
        print(f'Cjt:    {self.target.C* 1e15} fF')
        print(f'Ejt/Ect target: {np.real(self.target.Ej/self.target.Ec)}')
        print(f'w_target:    {self.omega_target * 1e-9 / constants.h} GHz')
        print(f'ng target:    {self.target.ng}')

        print(f'detunning:    {self.detunning * 1e-9 / constants.h} GHz')

        print(f'Cc:    {self.Cc * 1e15} fF')

    def init_operator(self):
        self.probe.init_operator()
        self.target.init_operator()

        self.I_cb = self.probe.I_cb  # Identity for qubit (charge basis)

        self.q_op_cb = self.probe.n_cb           # Charge operator (charge basis)
        self.e_iphi_op_cb = sp.sparse.diags(np.ones(2 * self.ncut, dtype=np.complex_), offsets=1)

        self.q1_q1 = qt.tensor((self.probe.n_cb + self.probe.ng_cb) * (self.probe.n_cb + self.probe.ng_cb), self.target.I_cb)
        self.q1_q2 = qt.tensor((self.probe.n_cb + self.probe.ng_cb), (self.target.n_cb + self.target.ng_cb))
        self.q2_q2 = qt.tensor(self.probe.I_cb, (self.target.n_cb + self.target.ng_cb) * (self.target.n_cb + self.target.ng_cb))

    ## GET KIN AND POT ##

    def get_kinetic_circuit(self):
        self.init_operator()

        C_mat = [
            [self.probe.C + self.Cc, -self.Cc],
            [-self.Cc, self.target.C + self.Cc]
        ]

        C_mat_inverse = np.linalg.inv(C_mat)
        
        kin = C_mat_inverse[0][0] * self.q1_q1
        kin += 2 * C_mat_inverse[0][1] * self.q1_q2
        kin += C_mat_inverse[1][1] * self.q2_q2

        kin *= 4 * 0.5 * constants.e **2

        return kin

    def get_potential_circuit(self):
        self.init_operator()

        potential = -0.5*self.probe.Ej * qt.tensor(self.probe.e_iphi_op_cb + self.probe.e_iphi_op_cb.trans(), self.target.I_cb)
        potential += -0.5*self.target.Ej * qt.tensor(self.probe.I_cb, self.target.e_iphi_op_cb + self.target.e_iphi_op_cb.trans())

        return potential

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

    def diagonalise_probe(self,update=False):
        if update:
            self.get_H_probe()
        else:
            try:
                self.Ham_probe
            except AttributeError:
                self.get_H_probe()
        
        self.eig_values_probe, self.eig_vectors_probe = self.H_probe.eigenstates(eigvals = 2)

        return self.eig_values_probe, self.eig_vectors_probe

    def diagonalise_target(self,update=False):
        if update:
            self.get_H_target()
        else:
            try:
                self.H_target
            except AttributeError:
                self.get_H_target()
        
        self.eig_values_target, self.eig_vectors_target = self.H_target.eigenstates(eigvals = 10)

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

        # self.state_p0eb = self.eig_vectors_probe[0].T
        # self.state_p1eb = self.eig_vectors_probe[1].T
        # self.state_target = []

        self.state_product_ebeb = []

        for i in range(len(self.eig_vectors_probe)):
            for j in range(len(self.eig_vectors_target)):
                self.state_product_ebeb.append(qt.tensor(self.eig_vectors_probe[i], self.eig_vectors_target[j]))

        # self.state_target_0_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        # self.state_target_0_cb[self.ncut] = 1
        # self.state_target_1_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        # self.state_target_1_cb[self.ncut + 1] = 1

        # c = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        # c[self.ncut] = 1
        # self.state_product_ebcb = []

        # for i in range(len(self.eig_vectors_probe)):
        #     for j in range(2 * self.ncut + 1):
        #         self.state_product_ebcb.append(sp.sparse.kron(self.eig_vectors_probe[i], c.T).T)

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
    
    ## CALCULATION FIRST TRY ##

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

        self.U_raw = self.H_circuit - qt.tensor(self.H_probe, self.target.I_cb) - qt.tensor(self.probe.I_cb, self.H_target)

        # size_ebcb = len(self.state_product_ebcb)
        # self.U_ebcb = np.zeros((size_ebcb, size_ebcb), dtype=object)

        # for i in range(size_ebcb):
        #     for j in range(size_ebcb):
        #         self.U_ebcb[i,j] = np.real(self.hc(self.state_product_ebcb[i]).dot(self.U_raw.dot(self.state_product_ebcb[j])).toarray()[0][0])
        
        size_ebeb = len(self.state_product_ebeb)
        self.U_ebeb = np.zeros((size_ebeb, size_ebeb), dtype=object)

        for i in range(size_ebeb):
            for j in range(size_ebeb):
                self.U_ebeb[i,j] = float(np.real((self.state_product_ebeb[i].dag() * self.U_raw * self.state_product_ebeb[j]).full()))
        self.U_ebeb = self.U_ebeb.astype(float)

        return self.U_ebeb, self.U_raw

    def extract_U_zz(self, update=False):
        if update:
            self.get_U_circuit(update=True)
        else:
            try:
                # self.U_ebcb
                self.U_ebeb
            except AttributeError:
                self.get_U_circuit()
        
        size_ebeb = len(self.state_product_ebeb)
        self.U_zz_ebeb = np.zeros((size_ebeb, size_ebeb), dtype=object)

        for i in range(size_ebeb):
            self.U_zz_ebeb[i,i] = self.U_ebeb[i,i]

        # size_ebcb = len(self.state_product_ebcb)
        # self.U_zz_ebcb = np.zeros((size_ebcb, size_ebcb), dtype=object)

        # for i in range(size_ebcb):
        #     self.U_zz_ebcb[i,i] = self.U_ebcb[i,i]
        self.U_zz_ebeb = self.U_zz_ebeb.astype(float)

        return self.U_zz_ebeb

    def extract_U_zx(self, update=False):
        if update:
            self.get_U_circuit(update=True)
        else:
            try:
                # self.U_ebcb
                self.U_ebeb
            except AttributeError:
                self.get_U_circuit()
        
        size_ebeb = len(self.state_product_ebeb)
        self.U_zx_ebeb = np.zeros((size_ebeb, size_ebeb), dtype=object)

        for i in range(size_ebeb):
            for j in range(size_ebeb):
                if i != j:
                    self.U_zx_ebeb[i,j] = self.U_ebeb[i,j]

        # size_ebcb = len(self.state_product_ebcb)
        # self.U_zx_ebcb = np.zeros((size_ebcb, size_ebcb), dtype=object)
        
        # for i in range(size_ebcb):
        #     for j in range(size_ebcb):
        #         if i != j:
        #             self.U_zx_ebcb[i,j] = self.U_ebcb[i,j]
        self.U_zx_ebeb = self.U_zx_ebeb.astype(float)

        return self.U_zx_ebeb

    def get_g_parr(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_product_ebeb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        U_cache = self.H_circuit - qt.tensor(self.H_probe, self.target.I_cb) - qt.tensor(self.probe.I_cb, self.H_target)
        self.g_parr = np.abs(np.real(qt.tensor(self.eig_vectors_probe[0], self.eig_vectors_target[0]).dag() * U_cache * qt.tensor(self.eig_vectors_probe[0], self.eig_vectors_target[0] )))
        self.g_parr = float(self.g_parr)

        return self.g_parr

    def get_g_perp(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_product_ebeb
            except AttributeError:
                self.init_qubit_states()

        U_cache = self.H_circuit - qt.tensor(self.H_probe, self.target.I_cb) - qt.tensor(self.I_cb, self.H_target)
        self.g_perp = np.abs(np.real(qt.tensor(self.eig_vectors_probe[0], self.eig_vectors_target[0]).dag() * U_cache * qt.tensor(self.eig_vectors_probe[1], self.eig_vectors_target[0])))
        self.g_perp = float(self.g_perp)

        return self.g_perp
    
    ## CALCULATION SECOND TRY ##

    def get_gparr2(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_product_ebeb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()
        
        self.state_probe_plus = 2**-0.5 * (self.eig_vectors_probe[0] + self.eig_vectors_probe[1])
        self.state_probe_minus = 2**-0.5 * (self.eig_vectors_probe[0] - self.eig_vectors_probe[1])
        self.state_target_plus = 2**-0.5 * (self.eig_vectors_target[0] + self.eig_vectors_target[1])
        self.state_target_minus = 2**-0.5 * (self.eig_vectors_target[0] - self.eig_vectors_target[1])

        self.state_product_ebeb_pp = qt.tensor(self.state_probe_plus, self.state_target_plus)
        # self.state_product_ebeb_pm = qt.tensor(self.state_probe_plus, self.state_target_minus)
        # self.state_product_ebeb_mp = qt.tensor(self.state_probe_minus, self.state_target_plus)
        self.state_product_ebeb_mm = qt.tensor(self.state_probe_minus, self.state_target_minus)

        g_parr = self.state_product_ebeb_pp.dag() * self.H_circuit * self.state_product_ebeb_mm

        g_parr = float(np.real(g_parr))

        return g_parr

    def get_gperp2(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_product_ebeb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()
        
        # self.state_probe_plus = 2**-0.5 * (self.eig_vectors_probe[0] + self.eig_vectors_probe[1])
        # self.state_probe_minus = 2**-0.5 * (self.eig_vectors_probe[0] - self.eig_vectors_probe[1])
        self.state_target_plus = 2**-0.5 * (self.eig_vectors_target[0] + self.eig_vectors_target[1])
        self.state_target_minus = 2**-0.5 * (self.eig_vectors_target[0] - self.eig_vectors_target[1])

        self.state_product_ebeb_0p = qt.tensor(self.eig_vectors_probe[0], self.state_target_plus)
        # self.state_product_ebeb_1p = qt.tensor(self.eig_vectors_probe[1], self.state_target_plus)
        # self.state_product_ebeb_0m = qt.tensor(self.eig_vectors_probe[0], self.state_target_minus)
        self.state_product_ebeb_1m = qt.tensor(self.eig_vectors_probe[1], self.state_target_minus)

        g_perp = self.state_product_ebeb_0p.dag() * self.H_circuit * self.state_product_ebeb_1m

        g_perp = float(np.real(g_perp))

        return g_perp
        
    ## CALCULATION IN CHARGE BASIS ##

    def get_U_cb(self, update=False): 
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.state_product_ebeb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

        self.U_cb = self.H_circuit - qt.tensor(self.H_probe, self.target.I_cb) - qt.tensor(self.probe.I_cb, self.H_target)
        self.U_cb = np.real(self.U_cb.full())

        return self.U_cb
    
    def get_g_parr_cb(self, update=False):
        if update:
            self.init_qubit_states(update=True)
            self.get_U_cb(update=True)
        else:
            try:
                self.state_product_ebeb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.U_cb
            except AttributeError:
                self.get_U_cb()
            
        self.state_target_cb_0 = qt.basis(2 * self.ncut + 1, 0)
        self.state_target_cb_1 = qt.basis(2 * self.ncut + 1, 1)
        self.state_probe_plus = 2**-0.5 * (self.eig_vectors_probe[0] + self.eig_vectors_probe[1])
        self.state_probe_minus = 2**-0.5 * (self.eig_vectors_probe[0] - self.eig_vectors_probe[1])

        self.state_product_ebeb_p0_ebcb = qt.tensor(self.state_probe_plus, self.state_target_cb_0)
        self.state_product_ebeb_p1_ebcb = qt.tensor(self.state_probe_plus, self.state_target_cb_1)
        self.state_product_ebeb_m0_ebcb = qt.tensor(self.state_probe_minus, self.state_target_cb_0)
        self.state_product_ebeb_m1_ebcb = qt.tensor(self.state_probe_minus, self.state_target_cb_1)

        self.g_parr_cb = self.state_product_ebeb_p1_ebcb.dag() * self.H_circuit * self.state_product_ebeb_m1_ebcb - self.state_product_ebeb_p0_ebcb.dag() * self.H_circuit * self.state_product_ebeb_m0_ebcb
        self.g_parr_cb = float(np.real(self.g_parr_cb))

        return self.g_parr_cb
    
    def get_g_perp_cb(self, update=False):
        if update:
            self.init_qubit_states(update=True)
            self.get_U_cb(update=True)
        else:
            try:
                self.state_product_ebeb
            except AttributeError:
                self.init_qubit_states()
            try:
                self.U_cb
            except AttributeError:
                self.get_U_cb()
        
        self.state_target_cb_0 = qt.basis(2 * self.ncut + 1, 0)
        self.state_target_cb_1 = qt.basis(2 * self.ncut + 1, 1)
        self.state_probe_eb_0 = self.eig_vectors_probe[0]
        self.state_probe_eb_1 = self.eig_vectors_probe[1]

        self.state_product_ebeb_00_ebcb = qt.tensor(self.state_probe_eb_0, self.state_target_cb_0)
        self.state_product_ebeb_01_ebcb = qt.tensor(self.state_probe_eb_0, self.state_target_cb_1)
        self.state_product_ebeb_10_ebcb = qt.tensor(self.state_probe_eb_1, self.state_target_cb_0)
        self.state_product_ebeb_11_ebcb = qt.tensor(self.state_probe_eb_1, self.state_target_cb_1)

        self.g_perp_cb = self.state_product_ebeb_11_ebcb.dag() * self.H_circuit * self.state_product_ebeb_01_ebcb - self.state_product_ebeb_10_ebcb.dag() * self.H_circuit * self.state_product_ebeb_00_ebcb
        self.g_perp_cb = float(np.real(self.g_perp_cb))

        return self.g_perp_cb    