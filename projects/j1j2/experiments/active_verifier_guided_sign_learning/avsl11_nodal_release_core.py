"""AVSL-11: exact FN nodal-release identity and walker-neighbor screening.

FN sign correction certificates require the EXACT positive old FN ground state;
the cheap screening function uses only root guide amplitudes and callable signs.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass
import numpy as np

def orbit_release_certificate(upper_edges_i,upper_edges_j,H_ij,orbit_index,
                              sign,guide,fn_phi,n_orbits=None):
    """Returns guaranteed FN energy improvement lower bound for each orbit.

    Same guide a on both sides. On crossing (i,j) with K=H_ij*s_i*s_j:
       G= sum K*(sqrt(a_j/a_i)*phi_i+sqrt(a_i/a_j)*phi_j)**2.
    Then E_FN(s,a)-E_FN(flip_orbit(s),a)>=G for exact old FN phi.
    """
    i=np.asarray(upper_edges_i,np.int32);j=np.asarray(upper_edges_j,np.int32)
    H=np.asarray(H_ij,float);orbit=np.asarray(orbit_index,np.int32)
    s=np.asarray(sign,float);a=np.asarray(guide,float);phi=np.asarray(fn_phi,float)
    if (a<=0).any() or len(phi)!=len(a) or abs(phi@phi-1)>1e-7:
        raise ValueError('Require normalized FN state and positive guide')
    if (H<0).any() or not np.all(np.abs(s)==1):
        raise ValueError('Require positive offdiagonal H and binary signs')
    crossing=orbit[i]!=orbit[j];i=i[crossing];j=j[crossing];H=H[crossing]
    K=H*s[i]*s[j]
    w=K*(np.sqrt(a[j]/a[i])*phi[i]+np.sqrt(a[i]/a[j])*phi[j])**2
    n=int(orbit.max()+1 if n_orbits is None else n_orbits)
    return (np.bincount(orbit[i],weights=w,minlength=n)+
            np.bincount(orbit[j],weights=w,minlength=n))

@dataclass
class NeighborBalance:
    score_by_orbit:dict
    n_unique_roots:int
    n_neighbor_inspections:int
    n_sign_queries:int
    n_guide_queries_on_candidates:int=0

def neighbor_sign_balance(mixed_roots,neighbors,current_sign,guide_at_root,
                          canonical_orbit):
    """Cheap ranker using callable s(x); NO candidate amplitude query.

    B(G) = [sum_{y root,x~y,x in G} c_y H_xy s_x s_y/a_y] /
           [sum_{y root,x~y,x in G} c_y H_xy/a_y].

    This estimates an observed sign-conflict FRACTION, not a Rayleigh gain,
    not a Krylov sign verifier, and not post-FN energy improvement. Candidate
    sign model calls/canonicalization need separate cost accounting.
    """
    counts=Counter(map(int,mixed_roots))
    support={canonical_orbit(x) for x in counts}
    num,den=defaultdict(float),defaultdict(float);seen=queried=0
    for y,c in counts.items():
        sy=int(current_sign(y));ay=float(guide_at_root(y))
        if sy not in (-1,1) or not np.isfinite(ay) or ay<=0:
            raise ValueError('Invalid root sign/guide')
        gy=canonical_orbit(y)
        for x,h in neighbors(y):
            seen+=1;x=int(x);h=float(h);g=canonical_orbit(x)
            if h<=0:raise ValueError('Nonpositive H_xy')
            if g==gy or g in support:continue
            sx=int(current_sign(x));queried+=1
            q=float(c)*h/ay
            num[g]+=q*sy*sx;den[g]+=q
    return NeighborBalance({k:num[k]/v for k,v in den.items()},len(counts),
                           seen,queried,0)
