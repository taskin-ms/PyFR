"""Standalone Python translation of Federrath's forcing_generator.F90.

Source: turbulence_generator/legacy_fortran/forcing_generator.F90
(Federrath et al. 2010, A&A 512, A81), sha256 9793ae2d...61d6267.

Routine by routine, in the original order, with the original arithmetic:

  ran1s, ran2      Numerical Recipes generators, integer recurrences as in
                   Fortran (32-bit, Schrage; no overflow occurs); ran2 keeps
                   its SAVEd shuffle table between calls
  st_grn           Box-Muller, cosine branch only, from two ran1s(st_seed)
  init_stir        integer k loops (0..256 per dimension, positive octant),
                   sign patterns (kx, +-ky, +-kz) incl. duplicates on the axes,
                   band / paraboloid / power-law amplitudes, st_solweightnorm;
                   power law: random angles from ran2 exactly as in Fortran,
                   including that ran2 is initialised with the EXPRESSION
                   -st_seed (a temporary) and then called with st_seed
  st_ounoiseinit   X0 = sigma Z, one Gaussian per OU variable, in order
  st_ounoiseupdate X_{n+1} = f X_n + sigma sqrt(1 - f^2) Z_n, f = exp(-dt/ts)
  st_calcPhases    Helmholtz projection with the original index arithmetic
  MAIN             set-up, then for step = 0..nsteps: update, calcPhases, write

Precision: 'double' reproduces the shipped build (-fdefault-real-8); 'single'
the default real(4) build. Transcendentals call the same glibc functions that
gfortran calls (log/cos/exp/acos/pow/sincos or their *f variants); integer
powers follow libgcc's __powidf2 and constant powers x**2.0 are x*x, as
gfortran compiles them; nint is lround (round half away from zero).

Only one equivalence-preserving shortcut is taken: the 0..256 integer loops
skip iterations whose |k| exceeds the upper bound by a margin, which have no
side effects in the Fortran (see _kloop).
"""
import ctypes
import ctypes.util
import math
import re
import struct

import numpy as np

_libm = ctypes.CDLL(ctypes.util.find_library('m') or 'libm.so.6')


def _fn(name, ct):
    f = getattr(_libm, name)
    f.restype = ct
    f.argtypes = [ct]*(2 if name.startswith('pow') else 1)
    return f


class Real:
    """Arithmetic of one Fortran real kind."""

    def __init__(self, precision='double'):
        self.precision = precision
        if precision == 'double':
            ct, sfx, self.R = ctypes.c_double, '', float
            self.nbytes, self.fmt = 8, 'd'
        elif precision == 'single':
            ct, sfx, self.R = ctypes.c_float, 'f', np.float32
            self.nbytes, self.fmt = 4, 'f'
        else:
            raise ValueError(precision)
        self._log, self._cos, self._exp = (_fn(n + sfx, ct) for n in ('log', 'cos', 'exp'))
        self._acos, self._pow = _fn('acos' + sfx, ct), _fn('pow' + sfx, ct)
        self._sincos = getattr(_libm, 'sincos' + sfx)
        self._sincos.restype = None
        self._sincos.argtypes = [ct, ctypes.POINTER(ct), ctypes.POINTER(ct)]
        self._ct = ct

    def log(self, x): return self.R(self._log(x))
    def cos(self, x): return self.R(self._cos(x))
    def exp(self, x): return self.R(self._exp(x))
    def acos(self, x): return self.R(self._acos(x))
    def pow(self, x, y): return self.R(self._pow(x, y))

    def sqrt(self, x):
        return self.R(math.sqrt(x)) if self.precision == 'double' else np.sqrt(self.R(x))

    def sincos(self, x):
        s, c = self._ct(), self._ct()
        self._sincos(x, ctypes.byref(s), ctypes.byref(c))
        return self.R(s.value), self.R(c.value)

    def powi(self, x, m):
        """libgcc __powidf2 / __powisf2 (integer exponent)."""
        x = self.R(x)
        n = -m if m < 0 else m
        y = x if n % 2 else self.R(1)
        n >>= 1
        while n:
            x = x*x
            if n % 2:
                y = y*x
            n >>= 1
        return self.R(1)/y if m < 0 else y

    @staticmethod
    def lround(x):
        """Fortran nint: round half away from zero (glibc lround)."""
        return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _idiv(a, b):
    """Fortran integer division (truncates towards zero)."""
    q = abs(a)//abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def read_inp(path):
    """The &input namelist of forcing_generator.inp (as Fortran reads it)."""
    txt = open(path).read()
    body = txt[txt.index('&input') + 6:txt.index('&end')]
    vals = {}
    for line in body.splitlines():
        line = line.split('!')[0].strip()
        if '=' in line:
            k, v = (s.strip() for s in line.split('=', 1))
            vals[k.lower()] = v
    return vals


class StirGenerator:
    """State and routines of module Stir_data plus MAIN's set-up."""

    st_maxmodes = 100000

    def __init__(self, inp, precision='double'):
        self.r = r = Real(precision)
        R = r.R
        if isinstance(inp, str):
            inp = read_inp(inp)
        ints = {'ndim', 'st_spectform', 'st_seed', 'nsteps_per_turnover_time'}
        g = {k: (int(v) if k in ints else R(float(v))) for k, v in inp.items()}
        self.inp = g
        self.ndim = g['ndim']
        self.xmin, self.xmax = g['xmin'], g['xmax']
        self.ymin, self.ymax = g['ymin'], g['ymax']
        self.zmin, self.zmax = g['zmin'], g['zmax']
        self.velocity = g['velocity']
        self.st_solweight = g['st_solweight']
        self.st_spectform = g['st_spectform']
        self.st_power_law_exp = g['st_power_law_exp']
        self.st_angles_exp = g['st_angles_exp']
        self.st_seed = g['st_seed']

        # twopi = 8 * atan(1.0), folded by the compiler (correctly rounded)
        self.twopi = R(8)*R(math.atan(1.0)) if precision == 'double' else \
            np.float32(8*np.float32(math.atan(1.0)))

        # ran2 SAVEd state: DATA idum2/123456789/, iv/NTAB*0/, iy/0/
        self._ran2_iv = [0]*33
        self._ran2_iy = 0
        self._ran2_idum2 = 123456789

        # MAIN set-up (forcing_generator.F90, lines 625-636)
        tiny = R(np.finfo(np.float64 if precision == 'double' else np.float32).tiny)
        Lx = self.xmax - self.xmin
        self.st_stirmin = (g['k_min'] - tiny)*self.twopi/Lx
        self.st_stirmax = (g['k_max'] + tiny)*self.twopi/Lx
        self.st_decay = Lx/g['k_driv']/self.velocity
        v = self.velocity
        self.st_energy = g['st_energy_coeff']*((v*v)*v)/Lx      # velocity**3
        self.end_time = g['end_time']*self.st_decay
        self.nsteps = g['nsteps_per_turnover_time']*r.lround(
            self.end_time/self.st_decay)

    # ------------------------------------------------------- generators
    def ran1s(self, idum):
        """Returns (value, idum after the call)."""
        R = self.r.R
        IA, IM, IQ, IR = 16807, 2147483647, 127773, 2836
        AM = R(1.0)/R(IM)
        EPS = R(1.2e-7)
        RNMX = R(1.0) - EPS
        if idum <= 0:
            idum = max(-idum, 1)
        k = _idiv(idum, IQ)
        idum = IA*(idum - k*IQ) - IR*k
        if idum < 0:
            idum += IM
        iy = idum
        return min(AM*R(iy), RNMX), idum

    def ran2(self, idum):
        """Returns (value, idum after the call); keeps the SAVEd state."""
        R = self.r.R
        IM1, IM2 = 2147483563, 2147483399
        AM = R(1.0)/R(IM1)
        IMM1 = IM1 - 1
        IA1, IA2, IQ1, IQ2, IR1, IR2 = 40014, 40692, 53668, 52774, 12211, 3791
        NTAB = 32
        NDIV = 1 + _idiv(IMM1, NTAB)
        EPS = R(1.2e-7)
        RNMX = R(1.0) - EPS
        iv = self._ran2_iv
        if idum <= 0:
            idum = max(-idum, 1)
            self._ran2_idum2 = idum
            for j in range(NTAB + 8, 0, -1):
                k = _idiv(idum, IQ1)
                idum = IA1*(idum - k*IQ1) - k*IR1
                if idum < 0:
                    idum += IM1
                if j <= NTAB:
                    iv[j] = idum
            self._ran2_iy = iv[1]
        k = _idiv(idum, IQ1)
        idum = IA1*(idum - k*IQ1) - k*IR1
        if idum < 0:
            idum += IM1
        idum2 = self._ran2_idum2
        k = _idiv(idum2, IQ2)
        idum2 = IA2*(idum2 - k*IQ2) - k*IR2
        if idum2 < 0:
            idum2 += IM2
        self._ran2_idum2 = idum2
        j = 1 + _idiv(self._ran2_iy, NDIV)
        iy = iv[j] - idum2
        iv[j] = idum
        if iy < 1:
            iy += IMM1
        self._ran2_iy = iy
        return min(AM*R(iy), RNMX), idum

    def st_grn(self):
        r = self.r
        r1, self.st_seed = self.ran1s(self.st_seed)
        r2, self.st_seed = self.ran1s(self.st_seed)
        return r.sqrt(r.R(2.0)*r.log(r.R(1.0)/r1))*r.cos(self.twopi*r2)

    # ---------------------------------------------------------- init_stir
    def _kloop(self, Lx, Ly, Lz):
        """The positive-octant loops 0..256 in Fortran order, yielding
        (kx, ky, kz, k) for |k| within [stirmin, stirmax]. Iterations are only
        skipped once a single component alone exceeds stirmax by a relative
        margin, so |k| > stirmax there and the Fortran does nothing."""
        r, R = self.r, self.r.R
        tp, hi = self.twopi, float(self.st_stirmax)*(1 + 1e-9)
        ikymax = 256 if self.ndim > 1 else 0
        ikzmax = 256 if self.ndim > 2 else 0
        for ikx in range(0, 257):
            kx = tp*R(ikx)/Lx
            if kx > hi:
                break
            for iky in range(0, ikymax + 1):
                ky = tp*R(iky)/Ly
                if ky > hi:
                    break
                for ikz in range(0, ikzmax + 1):
                    kz = tp*R(ikz)/Lz
                    if kz > hi:
                        break
                    k = r.sqrt((kx*kx + ky*ky) + kz*kz)
                    if k >= self.st_stirmin and k <= self.st_stirmax:
                        yield kx, ky, kz, k

    def init_stir(self):
        r, R = self.r, self.r.R
        ndim = self.ndim
        self.st_OUvar = r.sqrt(self.st_energy/self.st_decay)
        stirmin, stirmax = self.st_stirmin, self.st_stirmax
        parab_prefact = R(-4.0)/((stirmax - stirmin)*(stirmax - stirmin))
        kc = stirmin
        if self.st_spectform == 1:
            kc = R(0.5)*(stirmin + stirmax)
        sw = self.st_solweight
        one, two, three = R(1.0), R(2.0), R(3.0)
        pre = {3: r.sqrt(three/three)*r.sqrt(three)*one,
               2: r.sqrt(three/two)*r.sqrt(three)*one,
               1: r.sqrt(three/one)*r.sqrt(three)*one}[ndim]
        self.st_solweightnorm = pre/r.sqrt(one - two*sw + R(ndim)*(sw*sw))

        Lx, Ly, Lz = self.xmax - self.xmin, self.ymax - self.ymin, self.zmax - self.zmin
        dexp = R(ndim - 1)/R(2.0)                       # (ndim-1)/2.0
        self.st_tot_nmodes = sum(1 + (ndim > 1) + 2*(ndim > 2)
                                 for _ in self._kloop(Lx, Ly, Lz))
        modes, ampl, power = [], [], []
        maxm = self.st_maxmodes

        if self.st_spectform != 2:
            for kx, ky, kz, k in self._kloop(Lx, Ly, Lz):
                if len(modes) + 2**(ndim - 1) > maxm:
                    break               # Fortran 'exit' leaves the ikz loop
                if self.st_spectform == 0:
                    amplitude = one
                else:
                    amplitude = abs(parab_prefact*((k - kc)*(k - kc)) + one)
                amplitude = r.sqrt(amplitude)*r.pow(kc/k, dexp)
                modes.append((kx, ky, kz))
                ampl.append(amplitude)
                if ndim > 1:
                    modes.append((kx, -ky, kz))
                    ampl.append(amplitude)
                if ndim > 2:
                    modes.append((kx, ky, -kz))
                    ampl.append(amplitude)
                    modes.append((kx, -ky, -kz))
                    ampl.append(amplitude)
                power.append((k, (amplitude*amplitude)*r.powi(k/kc, ndim - 1),
                              amplitude, kx, ky, kz))
        else:
            # rand = ran2(-st_seed): the argument is an expression, so the
            # warm-up advances a temporary and st_seed itself is unchanged
            self.ran2(-self.st_seed)
            ikmin = max(1, r.lround(stirmin*Lx/self.twopi))
            ikmax = r.lround(stirmax*Lx/self.twopi)
            self.nang_per_k = []
            for ik in range(ikmin, ikmax + 1):
                nang = 2**ndim*int(math.ceil(r.pow(R(ik), self.st_angles_exp)))
                self.nang_per_k.append((ik, nang))
                for iang in range(1, nang + 1):
                    u, self.st_seed = self.ran2(self.st_seed)
                    phi = self.twopi*u
                    if ndim == 1:
                        if phi < self.twopi/R(2):
                            phi = R(0.0)
                        if phi >= self.twopi/R(2):
                            phi = self.twopi/R(2.0)
                    theta = self.twopi/R(4.0)
                    if ndim > 2:
                        u, self.st_seed = self.ran2(self.st_seed)
                        theta = r.acos(one - two*u)
                    u, self.st_seed = self.ran2(self.st_seed)
                    rand = (R(ik) + u) - R(0.5)
                    st, ct = r.sincos(theta)
                    sp, cp = r.sincos(phi)
                    kx = self.twopi*R(r.lround((rand*st)*cp))/Lx
                    ky = R(0.0)
                    if ndim > 1:
                        ky = self.twopi*R(r.lround((rand*st)*sp))/Ly
                    kz = R(0.0)
                    if ndim > 2:
                        kz = self.twopi*R(r.lround(rand*ct))/Lz
                    k = r.sqrt((kx*kx + ky*ky) + kz*kz)
                    if k >= stirmin and k <= stirmax:
                        if len(modes) + 2**(ndim - 1) > maxm:
                            break       # 'exit' leaves the angle loop
                        amplitude = r.pow(k/kc, self.st_power_law_exp)
                        fac = ((R(ik**(ndim - 1))/R(nang))*R(4.0))*r.sqrt(three)
                        amplitude = r.sqrt(amplitude*fac)*r.pow(kc/k, dexp)
                        modes.append((kx, ky, kz))
                        ampl.append(amplitude)
                        power.append((k, (amplitude*amplitude)*r.powi(k/kc, ndim - 1),
                                      amplitude, kx, ky, kz))

        self.st_nmodes = len(modes)
        self.st_mode = modes            # list of (kx, ky, kz)
        self.st_ampl = ampl
        self.power = power

    # ------------------------------------------------------------ OU
    def st_ounoiseinit(self, n, variance):
        return [self.st_grn()*variance for _ in range(n)]

    def st_ounoiseupdate(self, vector, variance, dt, ts):
        r, R = self.r, self.r.R
        f = r.exp(-dt/ts)
        c = variance*r.sqrt(R(1.0) - f*f)
        for i in range(len(vector)):
            g = self.st_grn()
            vector[i] = vector[i]*f + c*g
        return vector

    def st_calcPhases(self, ou):
        R = self.r.R
        sw = self.st_solweight
        one = R(1.0)
        aka, akb = [], []
        for i, m in enumerate(self.st_mode):
            ka = kb = kk = R(0.0)
            for j in range(self.ndim):
                kk = kk + m[j]*m[j]
                ka = ka + m[j]*ou[6*i + 2*j + 1]
                kb = kb + m[j]*ou[6*i + 2*j + 0]
            a, b = [R(0.0)]*3, [R(0.0)]*3
            for j in range(self.ndim):
                diva = m[j]*ka/kk
                divb = m[j]*kb/kk
                curla = ou[6*i + 2*j + 0] - divb
                curlb = ou[6*i + 2*j + 1] - diva
                a[j] = sw*curla + (one - sw)*divb
                b[j] = sw*curlb + (one - sw)*diva
            aka.append(a)
            akb.append(b)
        return aka, akb

    # ------------------------------------------------------------ MAIN
    def outfilename(self):
        ple = ''
        if self.st_spectform == 2:
            ple = '_pl' + f'{float(self.st_power_law_exp):+5.2f}'.strip()
        vel = f'{float(self.velocity):.2E}'
        sw = f'{float(self.st_solweight):3.1f}'
        return f'turb{ple}_v{vel}_zeta{sw}_seed{self.st_seed:06d}.dat'

    def run(self, nsteps=None, callback=None):
        """MAIN: init_stir, st_ounoiseinit, then for step = 0..nsteps
        st_ounoiseupdate, st_calcPhases and the record. Yields records."""
        R = self.r.R
        fname = self.outfilename()       # before init_stir changes st_seed
        self.init_stir()
        self.seed_after_init = self.st_seed
        ou = self.st_ounoiseinit(6*self.st_nmodes, self.st_OUvar)
        self.ou_init = list(ou)
        self.seed_after_ouinit = self.st_seed
        n = self.nsteps if nsteps is None else nsteps
        dt = self.end_time/R(self.nsteps)
        time = R(0.0)
        for step in range(0, n + 1):
            self.st_ounoiseupdate(ou, self.st_OUvar, dt, self.st_decay)
            aka, akb = self.st_calcPhases(ou)
            yield {'step': step, 'time': time, 'ou': list(ou), 'aka': aka,
                   'akb': akb, 'seed': self.st_seed}
            time = time + dt
        self._fname = fname

    # ------------------------------------------------------ file output
    def _rec(self, payload):
        n = len(payload)
        return struct.pack('<i', n) + payload + struct.pack('<i', n)

    def forcing_file_bytes(self):
        """The unformatted sequential file write_forcing_file produces."""
        f = '<' + self.r.fmt
        recs = []
        hdr = None
        for rec in self.run():
            if hdr is None:
                hdr = (struct.pack('<ii', self.nsteps, self.st_nmodes)
                       + struct.pack('<' + 7*self.r.fmt, self.end_time,
                                     self.st_decay, self.st_energy,
                                     self.st_solweight, self.velocity,
                                     self.st_stirmin, self.st_stirmax)
                       + struct.pack('<i', self.st_spectform))
                recs.append(self._rec(hdr))
            body = struct.pack('<i', rec['step']) + struct.pack(f, rec['time'])
            for arr in (self.st_mode, rec['aka'], rec['akb']):
                body += b''.join(struct.pack('<' + 3*self.r.fmt, *v) for v in arr)
            body += struct.pack('<' + self.r.fmt*len(self.st_ampl), *self.st_ampl)
            body += struct.pack('<' + self.r.fmt*len(rec['ou']), *rec['ou'])
            recs.append(self._rec(body))
        return b''.join(recs)

    def power_txt(self):
        """power.txt as written with '(6A16)' and '(6E16.6)'. As in the
        Fortran, the header line is only written for st_spectform /= 2 (the
        power-law branch appends rows to whatever power.txt exists)."""
        out = ''
        if self.st_spectform != 2:
            out = ''.join(f'{s:>16s}' for s in ('k', 'power', 'amplitude', 'kx', 'ky', 'kz')) + '\n'
        for row in self.power:
            out += ''.join(fortran_e(float(x), 16, 6) for x in row) + '\n'
        return out


def fortran_e(x, w, d):
    """Fortran Ew.d edit descriptor (0.ddddddE+xx), right-justified."""
    if x == 0:
        s = '0.' + '0'*d + 'E+00'
    else:
        m = f'{abs(x):.{d - 1}e}'
        digits, e = m.split('e')
        digits = digits.replace('.', '')
        e = int(e) + 1
        s = ('-' if x < 0 else '') + '0.' + digits + f'E{e:+03d}'
    return s.rjust(w)
