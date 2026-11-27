"""A Newton-Raphson AC power flow on the admittance matrices that pandapower builds, for long time series.

pandapower spends most of the time of a small power flow (about 20 ms on a 107-bus feeder) in data handling. The
time-series hosting analysis (feeder_ts.py) solves several million power flows, so this module takes the bus
admittance and branch matrices pandapower builds once and runs the same polar Newton-Raphson iteration (MATPOWER's
newtonpf, pandapower's default algorithm and tolerance) on them, with loads and generators as constant-power
injections. Results agree with pandapower to the solver tolerance; tests/test_feeder_ts.py and FastPF.validate()
check bus voltages and line and transformer loadings against pandapower on random operating points.

Scope (checked when the object is built): one external grid as slack, loads and static generators as constant
power (no voltage dependence), lines and two-winding transformers; no generators with voltage control, shunts, wards,
storage or three-winding transformers.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

UNSUPPORTED = ('gen', 'shunt', 'ward', 'xward', 'storage', 'trafo3w', 'impedance', 'dcline', 'motor', 'svc', 'tcsc', 'ssc', 'vsc')


class FastPF:
    def __init__(self, net, tol: float = 1e-8, max_iter: int = 20):
        import pandapower as pp
        for el in UNSUPPORTED:
            if el in net and len(net[el]) and ('in_service' not in net[el] or net[el]['in_service'].any()):
                raise ValueError(f'FastPF does not model {el} elements')
        if len(net.ext_grid) != 1:
            raise ValueError('FastPF needs exactly one external grid')
        for el in ('load',):
            for c in ('const_z_percent', 'const_i_percent', 'const_z_p_percent', 'const_i_p_percent', 'const_z_q_percent', 'const_i_q_percent'):
                if c in net[el] and (net[el][c].fillna(0) != 0).any():
                    raise ValueError('FastPF models loads as constant power only')
        self.net, self.tol, self.max_iter = net, tol, max_iter
        pp.runpp(net, numba=True)
        ppc, ppci = net._ppc, net._ppc['internal']
        # pandapower bus -> ppc bus (buses joined by closed switches share one) -> internal bus (isolated ppc buses,
        # e.g. behind open switches, are dropped and the rest renumbered in order)
        keep = ppc['bus'][:, 1].real != 4                                            # BUS_TYPE 4 = isolated
        if keep.sum() != ppci['bus'].shape[0]:
            raise ValueError('unexpected internal bus numbering')
        e2i = np.full(len(keep), -1); e2i[keep] = np.arange(keep.sum())
        self.branch_is = np.asarray(ppci['branch_is'], bool)
        self.n_branch = len(self.branch_is)
        self.base = float(ppci['baseMVA'])
        self.Y = sp.csr_matrix(ppci['Ybus']); self.Yf = sp.csr_matrix(ppci['Yf']); self.Yt = sp.csr_matrix(ppci['Yt'])
        self.ref = np.asarray(ppci['ref'], int); self.pv = np.asarray(ppci['pv'], int); self.pq = np.asarray(ppci['pq'], int)
        if len(self.pv):
            raise ValueError('FastPF has no voltage-controlled buses')
        self.pvpq = np.r_[self.pv, self.pq]
        self.lk = e2i[np.asarray(net._pd2ppc_lookups['bus'])]
        live = self.lk[net.bus.index[net.bus['in_service']].to_numpy()]
        self.lk_known = live >= 0
        self.nb = ppci['bus'].shape[0]
        self.base_kv = ppci['bus'][:, 9].astype(float)                               # BASE_KV
        self.V0 = np.asarray(ppci['V'], complex).copy()                               # pandapower solution, warm start
        self.Vref = self.V0[self.ref].copy()
        lo = net.load[net.load['in_service']]; sg = net.sgen[net.sgen['in_service']]
        self.load_rows, self.sgen_rows = lo.index.to_numpy(), sg.index.to_numpy()
        self.Cl = sp.csr_matrix((np.ones(len(lo)), (self.lk[lo['bus'].to_numpy()], np.arange(len(lo)))), shape=(self.nb, len(lo)))
        self.Cg = sp.csr_matrix((np.ones(len(sg)), (self.lk[sg['bus'].to_numpy()], np.arange(len(sg)))), shape=(self.nb, len(sg)))
        self.l_scale, self.g_scale = lo['scaling'].to_numpy(float), sg['scaling'].to_numpy(float)
        # branches: pandapower puts lines first, then two-winding transformers
        br = net._pd2ppc_lookups['branch']
        self.line_sl = slice(*br['line']) if 'line' in br else slice(0, 0)
        self.trafo_sl = slice(*br['trafo']) if 'trafo' in br else slice(0, 0)
        fb, tb = ppci['branch'][:, 0].real.astype(int), ppci['branch'][:, 1].real.astype(int)
        self.fkv, self.tkv = self.base_kv[fb], self.base_kv[tb]                      # internal (in-service) branches
        L = net.line
        self.line_imax = (L['max_i_ka'] * L['df'] * L['parallel']).to_numpy(float)
        self.line_on = L['in_service'].to_numpy(bool)
        T = net.trafo
        self.trafo_ihv = (T['sn_mva'] / (np.sqrt(3) * T['vn_hv_kv'])).to_numpy(float)
        self.trafo_ilv = (T['sn_mva'] / (np.sqrt(3) * T['vn_lv_kv'])).to_numpy(float)
        self.trafo_par = T['parallel'].to_numpy(float)
        self.trafo_on = T['in_service'].to_numpy(bool)
        # check the injection model against pandapower's own Sbus at the solved state
        S = self.sbus(lo['p_mw'].to_numpy(float), lo['q_mvar'].to_numpy(float), sg['p_mw'].to_numpy(float), sg['q_mvar'].to_numpy(float))
        nonref = np.setdiff1d(np.arange(self.nb), self.ref)                          # pandapower's Sbus holds the solved slack
        if not np.allclose(S[nonref], np.asarray(ppci['Sbus'])[nonref], atol=1e-10):
            raise ValueError('FastPF injection model differs from pandapower')

    # --------------------------------------------------------------------------------------------- model
    def sbus(self, p_load, q_load, p_sgen, q_sgen, extra_p_bus=None, extra_q_bus=None):
        """Complex bus injections in per unit: static generators minus loads (both scaled as in pandapower), minus
        any extra load given per pandapower bus index order of FastPF.lk (MW, MVAr)."""
        s = self.Cg @ ((p_sgen + 1j * q_sgen) * self.g_scale) - self.Cl @ ((p_load + 1j * q_load) * self.l_scale)
        if extra_p_bus is not None:
            s = s - extra_p_bus - (1j * extra_q_bus if extra_q_bus is not None else 0)
        return s / self.base

    def bus_vector(self, pd_bus_index: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Spread values given at pandapower buses onto the internal bus order (summing duplicates)."""
        idx = self.lk[np.asarray(pd_bus_index, int)]
        if (idx < 0).any():
            raise ValueError('a bus is isolated in the power flow model')
        out = np.zeros(self.nb)
        np.add.at(out, idx, values)
        return out

    def solve(self, S: np.ndarray, V0: np.ndarray | None = None):
        """Polar Newton-Raphson (MATPOWER newtonpf). Returns (V, converged). Small networks use dense algebra, which
        is faster than sparse algebra below a few hundred buses; the iteration is the same."""
        if self.nb <= 400:
            return self._solve_dense(S, V0)
        Y, pv, pq, pvpq = self.Y, self.pv, self.pq, self.pvpq
        V = (self.V0 if V0 is None else V0).copy()
        V[self.ref] = self.Vref
        Va, Vm = np.angle(V), np.abs(V)
        npvpq = len(pvpq)
        for _ in range(self.max_iter + 1):
            mis = V * np.conj(Y @ V) - S
            F = np.r_[mis[pvpq].real, mis[pq].imag]
            if np.max(np.abs(F)) < self.tol:
                return V, True
            Ibus = Y @ V
            dV = sp.diags(V); dI = sp.diags(Ibus); dVn = sp.diags(V / np.abs(V))
            dS_dVm = dV @ np.conj(Y @ dVn) + np.conj(dI) @ dVn
            dS_dVa = 1j * dV @ np.conj(dI - Y @ dV)
            J = sp.bmat([[dS_dVa[pvpq][:, pvpq].real, dS_dVm[pvpq][:, pq].real],
                         [dS_dVa[pq][:, pvpq].imag, dS_dVm[pq][:, pq].imag]], format='csc')
            dx = -spsolve(J, F)
            Va[pvpq] += dx[:npvpq]; Vm[pq] += dx[npvpq:]
            V = Vm * np.exp(1j * Va)
        return V, False

    def _solve_dense(self, S, V0):
        if not hasattr(self, '_Yd'):
            self._Yd = self.Y.toarray()
        Y, pq, pvpq = self._Yd, self.pq, self.pvpq
        V = (self.V0 if V0 is None else V0).copy()
        V[self.ref] = self.Vref
        Va, Vm = np.angle(V), np.abs(V)
        npvpq = len(pvpq)
        for _ in range(self.max_iter + 1):
            Ibus = Y @ V
            mis = V * np.conj(Ibus) - S
            F = np.r_[mis[pvpq].real, mis[pq].imag]
            if np.max(np.abs(F)) < self.tol:
                return V, True
            Vn = V / np.abs(V)
            dS_dVm = V[:, None] * np.conj(Y * Vn[None, :]); dS_dVm[np.diag_indices_from(dS_dVm)] += np.conj(Ibus) * Vn
            dS_dVa = -1j * V[:, None] * np.conj(Y * V[None, :]); dS_dVa[np.diag_indices_from(dS_dVa)] += 1j * V * np.conj(Ibus)
            J = np.block([[dS_dVa[np.ix_(pvpq, pvpq)].real, dS_dVm[np.ix_(pvpq, pq)].real],
                          [dS_dVa[np.ix_(pq, pvpq)].imag, dS_dVm[np.ix_(pq, pq)].imag]])
            dx = -np.linalg.solve(J, F)
            Va[pvpq] += dx[:npvpq]; Vm[pq] += dx[npvpq:]
            V = Vm * np.exp(1j * Va)
        return V, False

    def loadings(self, V):
        """Line and transformer loading in percent, as pandapower defines them (current based)."""
        k = self.base / np.sqrt(3)
        ifk, itk = np.zeros(self.n_branch), np.zeros(self.n_branch)                  # kA at each end, ppc branch order
        ifk[self.branch_is] = np.abs(self.Yf @ V) * k / self.fkv
        itk[self.branch_is] = np.abs(self.Yt @ V) * k / self.tkv
        ll = 100 * np.maximum(ifk[self.line_sl], itk[self.line_sl]) / self.line_imax
        ll = np.where(self.line_on, ll, 0.0)
        th = 100 * np.maximum(ifk[self.trafo_sl] / self.trafo_ihv, itk[self.trafo_sl] / self.trafo_ilv) / self.trafo_par
        th = np.where(self.trafo_on, th, 0.0)
        return ll, th

    # ---------------------------------------------------------------------------------------- validation
    def validate(self, n: int = 50, seed: int = 0, extra_mw: float = 2.0) -> dict:
        """Largest absolute differences from pandapower (voltage in pu, loadings in percentage points) over random
        operating points: loads and generators scaled at random and random extra load at random buses."""
        import pandapower as pp
        net = self.net
        rng = np.random.default_rng(seed)
        p0, q0 = net.load.loc[self.load_rows, 'p_mw'].to_numpy(float), net.load.loc[self.load_rows, 'q_mvar'].to_numpy(float)
        g0, h0 = net.sgen.loc[self.sgen_rows, 'p_mw'].to_numpy(float), net.sgen.loc[self.sgen_rows, 'q_mvar'].to_numpy(float)
        buses = net.load['bus'].unique()
        dv = dl = dt = 0.0
        tmp = pp.create_load(net, int(buses[0]), p_mw=0.0, q_mvar=0.0, name='fastpf validation')
        try:
            for _ in range(n):
                a, b = rng.uniform(0.2, 2.0, len(p0)), rng.uniform(0.0, 1.5, len(g0))
                eb = int(rng.choice(buses)); emw = float(rng.uniform(0, extra_mw))
                net.load.loc[self.load_rows, 'p_mw'] = p0 * a; net.load.loc[self.load_rows, 'q_mvar'] = q0 * a
                net.sgen.loc[self.sgen_rows, 'p_mw'] = g0 * b
                net.load.loc[tmp, ['bus', 'p_mw']] = [eb, emw]
                pp.runpp(net, numba=True, tolerance_mva=1e-10)
                S = self.sbus(p0 * a, q0 * a, g0 * b, h0, self.bus_vector([eb], [emw]))
                V, ok = self.solve(S)
                assert ok
                ll, th = self.loadings(V)
                ok_b = self.lk[net.bus.index] >= 0
                dv = max(dv, float(np.nanmax(np.abs(np.abs(V)[self.lk[net.bus.index][ok_b]] - net.res_bus['vm_pu'].to_numpy()[ok_b]))))
                dl = max(dl, float(np.max(np.abs(ll - net.res_line['loading_percent'].to_numpy()), initial=0.0)))
                dt = max(dt, float(np.max(np.abs(th - net.res_trafo['loading_percent'].to_numpy()), initial=0.0)))
        finally:
            net.load.drop(tmp, inplace=True)
            net.load.loc[self.load_rows, 'p_mw'] = p0; net.load.loc[self.load_rows, 'q_mvar'] = q0
            net.sgen.loc[self.sgen_rows, 'p_mw'] = g0
        return {'max_abs_vm_pu': dv, 'max_abs_line_loading_pct': dl, 'max_abs_trafo_loading_pct': dt, 'points': n}
