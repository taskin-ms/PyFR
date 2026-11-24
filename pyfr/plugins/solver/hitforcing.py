import json
import math

import numpy as np

from pyfr.cache import memoize
from pyfr.mpiutil import get_comm_rank_root, mpi
from pyfr.plugins.common import init_csv
from pyfr.plugins.solver.base import BaseSolverPlugin
from pyfr.plugins.solver.federrath_stir import StirGenerator
from pyfr.quadrules import get_quadrule


class FederrathCoeffs:
    # Federrath's forcing_generator.F90 (federrath_stir.py, a bitwise-verified
    # port) as the source of the Fourier coefficients. Records follow MAIN:
    # init_stir, st_ounoiseinit, then per record st_ounoiseupdate and
    # st_calcPhases, record 0 being the state after the first update. As in
    # the FLASH reader of that file, record n = floor(t/dt) is held constant
    # over [n dt, (n + 1) dt) and the acceleration is
    #   a(x) = st_solweightnorm sum_m 2 st_ampl_m (aka_m cos k.x - akb_m sin k.x)
    #        = sum_m Re(fhat_m e^{i k.x}),  fhat_m = 2 swn ampl_m (aka_m + i akb_m)
    # The recursive OU state lives on the root rank only.
    VERSION = 1

    # forcing_generator.inp keys and the defaults of the shipped file
    KEYS = {'velocity': '1.0', 'k_driv': '2.0', 'k_min': '1.0',
            'k_max': '3.0', 'st_spectform': '1', 'st_power_law_exp': '1.5',
            'st_angles_exp': '1.0', 'st_energy_coeff': '5.0e-3',
            'st_seed': '140281', 'end_time': '10.0',
            'nsteps_per_turnover_time': '10'}

    def __init__(self, inp, precision, boxlen):
        self.inp, self.precision = dict(inp), precision
        self.gen = g = StirGenerator(inp, precision)
        g.init_stir()
        self.seed_init = g.st_seed

        # MAIN: dt = end_time / nsteps
        self.dt = g.end_time/g.r.R(g.nsteps)
        self.dtk = float(self.dt)

        # Integer wave vectors for the kernel, k = 2 pi n / box-length
        for lo, hi in ((g.xmin, g.xmax), (g.ymin, g.ymax), (g.zmin, g.zmax)):
            if abs(float(hi - lo) - boxlen) > 1e-12*boxlen:
                raise ValueError('federrath: xmax - xmin etc. must equal '
                                 'box-length (cubic periodic box)')
        n = np.array(g.st_mode, dtype=float)*boxlen/(2*math.pi)
        self.modes = np.rint(n).astype(int)
        if np.abs(n - self.modes).max() > 1e-9:
            raise ValueError('federrath: modes are not integer wave vectors')

        self.scale = (2.0*float(g.st_solweightnorm)
                      * np.array(g.st_ampl, dtype=float))
        self._cache = {}
        self._head = None

    def _fhat(self, aka, akb):
        return self.scale[:, None]*(np.array(aka, dtype=float)
                                    + 1j*np.array(akb, dtype=float))

    def _from_start(self):
        g = self.gen
        g.st_seed = self.seed_init
        ou = g.st_ounoiseinit(6*g.st_nmodes, g.st_OUvar)
        self._head = (-1, g.st_seed, ou)
        self._cache = {}

    def _advance(self, n):
        # Records are produced strictly in order; going back means a replay
        if self._head is None or (n < self._head[0] and n not in self._cache):
            self._from_start()
        g = self.gen
        step, g.st_seed, ou = self._head
        ou = list(ou)
        while step < n:
            g.st_ounoiseupdate(ou, g.st_OUvar, self.dt, g.st_decay)
            step += 1
            aka, akb = g.st_calcPhases(ou)
            self._cache[step] = (self._fhat(aka, akb), g.st_seed, list(ou))
        self._head = (step, g.st_seed, ou)

    def record(self, n):
        if n not in self._cache:
            self._advance(n)
        return self._cache[n][0]

    def window(self, n0, nrow):
        # Rows hold [A_n, 0]: constant within each record
        nm = len(self.modes)
        buf = np.zeros((nrow, 12*nm))
        for j in range(nrow):
            a = self.record(n0 + j)
            buf[j, :6*nm] = np.stack([a.real, a.imag], axis=-1).ravel()

        return buf

    def evict(self, nmin):
        for n in [n for n in self._cache if n < nmin]:
            del self._cache[n]

    def fingerprint(self):
        return {'inp': self.inp, 'precision': self.precision,
                'nmodes': len(self.modes)}

    def state(self, n):
        # OU vector and ran1s seed after record n: everything the recursion
        # needs (ran2 is only used inside init_stir)
        self.record(n)
        _, seed, ou = self._cache[n]
        return {'version': self.VERSION, 'record': n, 'seed': seed,
                'ou': [float(x).hex() for x in ou], **self.fingerprint()}

    def restore(self, doc):
        if doc.get('version') != self.VERSION:
            raise RuntimeError('federrath: incompatible forcing state')
        for k, v in self.fingerprint().items():
            if doc.get(k) != v:
                raise RuntimeError(f'federrath: {k} changed across restart; '
                                   'the stochastic sequence would not continue')
        g = self.gen
        R = g.r.R
        ou = [R(float.fromhex(h)) for h in doc['ou']]
        n = doc['record']
        aka, akb = g.st_calcPhases(ou)
        self._cache = {n: (self._fhat(aka, akb), doc['seed'], list(ou))}
        self._head = (n, doc['seed'], ou)


class HITForcingWork:
    # Volume averages of the energy-equation work term rho*u.f and of rho*E,
    # integrated with the solution point quadrature (the norm in which the
    # scheme conserves) and reduced per element on the backend
    def __init__(self, intg, tplargs, extrns):
        self.backend = backend = intg.backend
        self._ele_banks = intg.system.ele_banks
        self._extrns = extrns

        backend.pointwise.register(
            'pyfr.plugins.solver.kernels.hitforcingwork'
        )

        # The macro is rendered with the work term but not the sink so that
        # it forms rho*u.f exactly as the source kernel does
        efrst = next(iter(intg.system.ele_map.values()))
        self._tplargs = tplargs | {'hitf_work': True, 'hitf_sink': False,
                                   'nvars': efrst.nvars}

        self._edata, vol = [], 0.0
        for ename, eles in intg.system.ele_map.items():
            rname = intg.cfg.get(f'solver-elements-{ename}', 'soln-pts')
            r = get_quadrule(ename, rname, eles.nupts)

            m0 = eles.shape.ubasis.nodal_basis_at(r.pts)
            if not np.allclose(m0, np.eye(eles.nupts)):
                raise ValueError('energy = work-sink requires solution '
                                 'points that form a quadrature rule')

            wts = r.wts[:, None] / eles.rcpdjac_at_np('upts')
            vol += wts.sum()

            self._edata.append((
                [eles.nupts, eles.neles], eles.ploc_at('upts'),
                backend.const_matrix(wts.astype(backend.fpdtype)),
                backend.matrix((2, eles.neles), tags={'align'})
            ))

        comm, rank, root = get_comm_rank_root()
        self._vol = comm.allreduce(vol, op=mpi.SUM)

    @memoize
    def _get_kerns(self, uidx):
        kerns = []
        for (dims, ploc, wts, out), banks in zip(self._edata, self._ele_banks):
            kerns.append(self.backend.pointwise.hitforcingwork(
                tplargs=self._tplargs, dims=dims,
                extrns={k: s for k, (s, v) in self._extrns.items()},
                u=banks[uidx], ploc=ploc, wts=wts, out=out,
                **{k: v for k, (s, v) in self._extrns.items()}
            ))

        return kerns

    def __call__(self, intg):
        kerns = self._get_kerns(intg.idxcurr)
        for k in kerns:
            k.bind(t=intg.tcurr)
        self.backend.run_kernels(kerns)

        local = sum(out.get().sum(axis=1) for *_, out in self._edata)

        comm, rank, root = get_comm_rank_root()
        comm.Allreduce(mpi.IN_PLACE, local, op=mpi.SUM)

        pin, emean = local / self._vol
        return float(pin), float(emean)


class HITForcingPlugin(BaseSolverPlugin):
    name = 'hitforcing'
    systems = 'euler|navier-stokes'
    dimensions = '3'

    def __init__(self, intg, cfgsect):
        super().__init__(intg, cfgsect)

        cfg, sect = self.cfg, cfgsect

        energy = cfg.get(sect, 'energy', 'none')
        boxlen = cfg.getfloat(sect, 'box-length', 2*math.pi)
        if energy not in {'none', 'work', 'work-sink'}:
            raise ValueError('Invalid energy source; must be none, work or '
                             'work-sink')

        # Federrath's forcing_generator.F90 sequence; keys as in its
        # forcing_generator.inp, defaults those of the shipped file
        inp = {k: cfg.get(sect, k, v) for k, v in FederrathCoeffs.KEYS.items()}
        inp['ndim'] = '3'
        inp['st_solweight'] = cfg.get(sect, 'st_solweight', '0.5')
        for c in 'xyz':
            inp[f'{c}min'] = cfg.get(sect, f'{c}min', '0.0')
            inp[f'{c}max'] = cfg.get(sect, f'{c}max', repr(boxlen))
        prec = cfg.get(sect, 'federrath-precision', 'double')
        self.coeffs = FederrathCoeffs(inp, prec, boxlen)
        self.dtk = dtk = self.coeffs.dtk
        nmodes = len(self.coeffs.modes)

        # Knot rows needed to cover all stage times of one step
        dtmax = getattr(intg, 'dtmax', float('inf'))
        dtref = dtmax if math.isfinite(dtmax) else intg.dt
        nrow = max(2, math.ceil(2*dtref / dtk) + 1)
        self.nrow = nrow = cfg.getint(sect, 'knot-window', nrow)

        # Per-knot coefficients in single precision; window start in full
        backend = intg.backend
        self._coef = backend.matrix((nrow, 12*nmodes), dtype=np.float32)
        self._t0 = backend.matrix((1, 1))
        self._n0 = None

        # The sequence is recursive, so the window is filled in setup() once a
        # checkpointed state, if any, has been restored (on restart plugins
        # are not called before the first step)
        self._tinit = (intg.tcurr, intg.dt)

        # Momentum source rho*f with the local density, as FLASH applies it
        tplargs = {
            'hitf_modes': self.coeffs.modes.tolist(), 'hitf_nrow': nrow,
            'hitf_rdtk': 1 / dtk, 'hitf_kscale': 2*math.pi / boxlen,
            'hitf_work': energy != 'none', 'hitf_sink': energy == 'work-sink',
            'hitf_rhoref': None
        }

        extrns = {
            'hitf_c': (f'in broadcast float[{nrow}][{12*nmodes}]', self._coef),
            'hitf_t0': ('in broadcast fpdtype_t[1][1]', self._t0)
        }

        # Homogeneous sink Lambda = <rho u.f>_V, measured from the solution
        self._sink = energy == 'work-sink'
        if self._sink:
            self._lam = backend.matrix((1, 1))
            self._work = HITForcingWork(intg, tplargs, dict(extrns))
            self._tprev = None
            self._init_stats(intg)

            # Lambda can only be measured once the backend is committed;
            # fresh runs then call plugins, restarts do not, so evaluate it
            # ahead of the first RHS evaluation instead
            def first_rhs(*args):
                del intg.system.rhs
                self._update_sink(intg)
                intg.system.rhs(*args)

            intg.system.rhs = first_rhs

            extrns['hitf_lam'] = ('in broadcast fpdtype_t[1][1]', self._lam)

        for eles in intg.system.ele_map.values():
            eles.add_src_macro('pyfr.plugins.solver.kernels.hitforcing',
                               'hitforcing', tplargs, ploc=True, soln=True)
            for name, (spec, value) in extrns.items():
                eles.set_external(name, spec, value=value)

    def setup(self, sdata, prevcfg, serialiser):
        # Continue the recursive sequence from the checkpointed OU vector and
        # ran1s seed; without one (fresh run, or a file written without it)
        # the sequence is replayed deterministically from st_seed
        comm, rank, root = get_comm_rank_root()
        if rank == root and sdata is not None:
            self.coeffs.restore(json.loads(bytes(np.asarray(sdata)).decode()))
        self._update(*self._tinit)

        def datafn():
            doc = self.coeffs.state(self._n0)
            return np.frombuffer(json.dumps(doc).encode(), dtype=np.uint8)

        serialiser.register(self.sprefix, datafn if rank == root else None)

    def __call__(self, intg):
        self._update(intg.tcurr, intg.dt)

        if self._sink:
            self._update_sink(intg)

    def _init_stats(self, intg):
        comm, rank, root = get_comm_rank_root()

        # Optional record of P_in, the Lambda applied over the preceding
        # step, <rho E>_V and the residual of the mean energy budget
        if rank == root and self.cfg.hasopt(self.cfgsect, 'file'):
            self._stats = init_csv(self.cfg, self.cfgsect,
                                   't,p_in,lambda,e_mean,residual', nflush=1,
                                   isrestart=intg.isrestart)
        else:
            self._stats = None

    def _update_sink(self, intg):
        t = intg.tcurr

        # Plugins can be called more than once at the same time
        if self._tprev is not None and t - self._tprev <= self.tol:
            return

        pin, emean = self._work(intg)

        if self._stats:
            if self._tprev is None:
                lam = res = float('nan')
            else:
                # d<rho E>/dt - (P_in - Lambda) over the step, with the
                # trapezoidal mean of P_in and Lambda = P_in(t_n-1)
                tp, lam, ep = self._tprev, self._pprev, self._eprev
                res = (emean - ep) / (t - tp) - 0.5*(pin - lam)

            self._stats(t, pin, lam, emean, res)

        # Hold Lambda = P_in(t_n) over the step t_n -> t_n+1
        self._lam.set(np.array([[pin]]))
        self._tprev, self._pprev, self._eprev = t, pin, emean

    def _update(self, tcurr, dt):
        n0 = math.floor(tcurr / self.dtk)

        if math.floor((tcurr + dt) / self.dtk) - n0 > self.nrow - 1:
            raise RuntimeError('Time step spans too many forcing records; '
                               'increase knot-window')

        # Coefficients only change when the window moves
        if n0 != self._n0:
            # Computed on the root rank so all ranks see identical bits
            comm, rank, root = get_comm_rank_root()
            if rank == root:
                buf = self.coeffs.window(n0, self.nrow)
            else:
                buf = np.empty((self.nrow, 12*len(self.coeffs.modes)))
            comm.Bcast(buf, root=root)

            self._coef.set(buf)
            self._t0.set(np.array([[n0*self.dtk]]))
            self.coeffs.evict(n0)
            self._n0 = n0
