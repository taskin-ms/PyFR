<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%
    nm = len(hitf_modes)
    kmax = max(abs(c) for k in hitf_modes for c in k)

    def fl(v):
        return repr(float(v)) + 'f'

    # Complex factor e^{i*k_d*x_d} as (re, im) expressions
    def efac(d, k):
        re, im = f'e{d}{abs(k)}r', f'e{d}{abs(k)}i'
        return (re, im) if k > 0 else (re, f'(-{im})')
%>

<%pyfr:macro name='hitforcing' params='t, u, ploc, src' externs='hitf_c, hitf_t0, hitf_lam'>
    // Locate t within the knot window; done in fpdtype_t as t may be large
    fpdtype_t s = (t - hitf_t0[0][0])*${hitf_rdtk};
    int j = (int) s;
    j = (j < 0) ? 0 : ((j > ${hitf_nrow - 1}) ? ${hitf_nrow - 1} : j);
    float w = (float) (s - j);
    w = (w < 0.0f) ? 0.0f : ((w > 1.0f) ? 1.0f : w);

    // e^{i*n*x_d} for n = 1 .. kmax via angle addition from one sin/cos
% for d in range(3):
    float xs${d} = ${fl(hitf_kscale)}*((float) ploc[${d}]);
    float e${d}1r = cos(xs${d}), e${d}1i = sin(xs${d});
  % for n in range(2, kmax + 1):
    float e${d}${n}r = e${d}${n - 1}r*e${d}1r - e${d}${n - 1}i*e${d}1i;
    float e${d}${n}i = e${d}${n - 1}r*e${d}1i + e${d}${n - 1}i*e${d}1r;
  % endfor
% endfor

    float er, ei, tr, fr, fi;
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

% for m, k in enumerate(hitf_modes):
<%
    facs = [efac(d, kd) for d, kd in enumerate(k) if kd]
%>
    // k = ${tuple(k)}
    er = ${facs[0][0]};
    ei = ${facs[0][1]};
  % for fre, fim in facs[1:]:
    tr = er*${fre} - ei*${fim};
    ei = er*${fim} + ei*${fre};
    er = tr;
  % endfor
  % for c, fv in enumerate(['fx', 'fy', 'fz']):
    fr = hitf_c[j][${6*m + 2*c}] + w*hitf_c[j][${6*nm + 6*m + 2*c}];
    fi = hitf_c[j][${6*m + 2*c + 1}] + w*hitf_c[j][${6*nm + 6*m + 2*c + 1}];
    ${fv} += fr*er - fi*ei;
  % endfor
% endfor

    fpdtype_t rho = ${'u[0]' if hitf_rhoref is None else hitf_rhoref};
    src[1] += rho*((fpdtype_t) fx);
    src[2] += rho*((fpdtype_t) fy);
    src[3] += rho*((fpdtype_t) fz);
% if hitf_sink:
    src[4] += u[1]*((fpdtype_t) fx) + u[2]*((fpdtype_t) fy) + u[3]*((fpdtype_t) fz) - hitf_lam[0][0];
% elif hitf_work:
    src[4] += u[1]*((fpdtype_t) fx) + u[2]*((fpdtype_t) fy) + u[3]*((fpdtype_t) fz);
% endif
</%pyfr:macro>
