
        ## OLD ##
        # self.probe_0_eb = self.eig_vectors_probe[0].T
        # self.probe_1_eb = self.eig_vectors_target[1].T
        # self.probe_plus_eb = 2**-0.5 * (self.probe_0_eb + self.probe_1_eb)
        # self.probe_minus_eb = 2**-0.5 * (self.probe_0_eb - self.probe_1_eb)
        # self.probe_jplus_eb = 2**-0.5 * (self.probe_0_eb + 1j * self.probe_1_eb)
        # self.probe_jminus_eb = 2**-0.5 * (self.probe_0_eb - 1j * self.probe_1_eb)

        # self.target_0_eb = self.eig_vectors_target[0].T
        # self.target_1_eb = self.eig_vectors_target[1].T
        # self.target_plus_eb = 2**-0.5 * (self.target_0_eb + self.target_1_eb)
        # self.target_minus_eb = 2**-0.5 * (self.target_0_eb - self.target_1_eb)
        # self.target_jplus_eb = 2**-0.5 * (self.target_0_eb + 1j * self.target_1_eb)
        # self.target_jminus_eb = 2**-0.5 * (self.target_0_eb - 1j * self.target_1_eb)

        # self.target_0_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        # self.target_0_cb[self.ncut] = 1
        # self.target_1_cb = sp.sparse.csr_matrix((2 * self.ncut + 1,1))
        # self.target_1_cb[self.ncut + 1] = 1
## OLD CODE ##
    def init_prod_states(self, update=False):
        if update:
            self.init_qubit_states(update=True)
        else:
            try:
                self.probe_0_eb
                self.target_0_eb
            except AttributeError:
                self.init_qubit_states(update=True)

        self.eb_eb_00 = sp.sparse.kron(self.probe_0_eb, self.target_0_eb)
        self.eb_eb_01 = sp.sparse.kron(self.probe_0_eb, self.target_1_eb)
        self.eb_eb_10 = sp.sparse.kron(self.probe_1_eb, self.target_0_eb)
        self.eb_eb_11 = sp.sparse.kron(self.probe_1_eb, self.target_1_eb)

        self.eb_cb_00 = sp.sparse.kron(self.probe_0_eb, self.target_0_cb)
        self.eb_cb_01 = sp.sparse.kron(self.probe_0_eb, self.target_1_cb)
        self.eb_cb_10 = sp.sparse.kron(self.probe_1_eb, self.target_0_cb)
        self.eb_cb_11 = sp.sparse.kron(self.probe_1_eb, self.target_1_cb)
        self.eb_cb_p0 = sp.sparse.kron(self.probe_plus_eb, self.target_0_cb)
        self.eb_cb_p1 = sp.sparse.kron(self.probe_plus_eb, self.target_1_cb)
        self.eb_cb_m0 = sp.sparse.kron(self.probe_minus_eb, self.target_0_cb)
        self.eb_cb_m1 = sp.sparse.kron(self.probe_minus_eb, self.target_1_cb)

    def calc_delta_probe(self, update=False):
        if update:
            self.get_H_circuit()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

            try:
                self.eb_eb_10
            except AttributeError:
                self.init_prod_states()
        
        delta_p = (self.hc(self.eb_eb_10).dot(self.Ham.dot(self.eb_eb_10)) - self.hc(self.eb_eb_00).dot(self.Ham.dot(self.eb_eb_00))).toarray()[0][0]

        return delta_p
        
    def calc_delta_target(self, update=False):
        if update:
            self.get_H_circuit()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.H()

            try:
                self.eb_eb_01
            except AttributeError:
                self.init_prod_states()

        delta_t = (self.hc(self.eb_eb_01).dot(self.Ham.dot(self.eb_eb_01)) - self.hc(self.eb_eb_00).dot(self.Ham.dot(self.eb_eb_00))).toarray()[0][0]

        return delta_t

    def calc_g_parr(self, update=False):
        if update:
            self.get_H_circuit()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

            try:
                self.eb_cb_p1
            except AttributeError:
                self.init_prod_states()

        g_parr = self.hc(self.eb_cb_p1).dot(self.Ham.dot(self.eb_cb_m1)).toarray()[0][0] - self.hc(self.eb_cb_p0).dot(self.Ham.dot(self.eb_cb_m0)).toarray()[0][0]

        return g_parr

    def calc_g_perp(self, update=False):
        if update:
            self.get_H_circuit()
            self.init_prod_states(update=True)
        else:
            try:
                self.H_circuit
            except AttributeError:
                self.get_H_circuit()

            try:
                self.eb_cb_11
            except AttributeError:
                self.init_prod_states()

        g_perp = self.hc(self.eb_cb_11).dot(self.Ham.dot(self.eb_cb_01)).toarray()[0][0] - self.hc(self.eb_cb_10).dot(self.Ham.dot(self.eb_cb_00)).toarray()[0][0]

        return g_perp