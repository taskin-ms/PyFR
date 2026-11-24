<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.plugins.solver.kernels.hitforcing'/>

<%pyfr:kernel name='hitforcingwork' ndim='2'
              t='scalar fpdtype_t'
              u='in fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[3]'
              wts='in fpdtype_t'
              out='out broadcast-col reduce(sum) fpdtype_t[2]'>
    // Energy source of the work term, formed exactly as in the RHS
    fpdtype_t src[${nvars}] = {};
    ${pyfr.expand('hitforcing', 't', 'u', 'ploc', 'src')};

    out[0] = wts*src[4];
    out[1] = wts*u[4];
</%pyfr:kernel>
