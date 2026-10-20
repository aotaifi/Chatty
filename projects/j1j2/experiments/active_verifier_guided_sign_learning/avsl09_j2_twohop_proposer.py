#!/usr/bin/env python3
"""AVSL-09: geometry-only two-hop candidate proposal for 20-site J1-J2 (J2=0.5).

For use with the *existing* tilted20 BONDS from avsl_cooperative20.py:
    BONDS = [(two_site_bitmask, H_offdiagonal), ...]
    H_nn = 0.5, H_nnn = 0.25.

The proposer NEVER calls FN amplitudes, sign networks or Krylov. A target
is not an accepted correction: it must pass the actual callable Krylov
verifier, global sign-consistency checks and (later) FN reprojection.

mode='j2_strict' uses only legal J2 exchanges. On a fully Néel state,
where no J2 exchange exists, it returns the state unchanged, with
blocked=True; it NEVER falls back to an unrestricted bond. Both arms
can suppress a direct reversal on hop 2 for an identical fair control.
"""
from dataclasses import dataclass

@dataclass
class Proposal:
    endpoint: int
    intermediate: int
    attempted_hops: int
    inspected_bonds: int
    n_legal_choices: int
    blocked: bool


def _legal(x, bonds, mode, j2_half=.25):
    out=[]
    for bits,h in bonds:
        bits=int(bits)
        if mode=='j2_strict' and abs(float(h)-float(j2_half))>1e-12:
            continue
        if (int(x)&bits).bit_count()==1:
            out.append(int(x)^bits)
    return out


def two_hop(x, bonds, rng, mode='all', nonbacktrack=True, j2_half=.25):
    """Return one two-hop configuration and transparent geometric query count.

    Both 'all' and 'j2_strict' enumerate the SAME full bond table on each
    performed hop. The first hop never uses amplitude or energy information.
    The second hop excludes the exact first root when possible.
    """
    if mode not in ('all','j2_strict'):
        raise ValueError(mode)
    x=int(x)
    y=x
    attempts=0
    n_choices=0
    scanned=0
    for step in range(2):
        opts=_legal(y,bonds,mode,j2_half)
        scanned+=len(bonds)
        n_choices+=len(opts)
        if nonbacktrack and step==1:
            fresh=[z for z in opts if z!=x]
            if fresh:
                opts=fresh
        if not opts:
            return Proposal(y,y,attempts,scanned,n_choices,True)
        z=int(opts[int(rng.integers(len(opts)))])
        if step==0:intermediate=z
        y=z
        attempts+=1
    return Proposal(y,intermediate,attempts,scanned,n_choices,False)


def j2_two_hop(x,bonds,rng,nonbacktrack=True):
    return two_hop(x,bonds,rng,mode='j2_strict',nonbacktrack=nonbacktrack)


def random_two_hop(x,bonds,rng,nonbacktrack=True):
    return two_hop(x,bonds,rng,mode='all',nonbacktrack=nonbacktrack)
