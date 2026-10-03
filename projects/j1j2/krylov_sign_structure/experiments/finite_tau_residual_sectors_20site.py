#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import scipy.sparse.csgraph as cs
from finite_tau_matching_20site_exact import build_H, D, special_columns

ROOT=Path("/Users/aliotaifi/Chatty/projects/j1j2/krylov_sign_structure")

def endpoint_sign(H, diag, y):
    A=(H-sp.diags(diag)).copy()
    A.data=np.ones_like(A.data)
    dist=np.asarray(cs.shortest_path(A,directed=False,unweighted=True,indices=[y]))[0]
    return np.where((dist.astype(np.int64)%2)==0,1.0,-1.0)

def split_gauged(H, diag, s):
    co=H.tocoo(); off=co.row!=co.col
    r=co.row[off].astype(np.int32)
    c=co.col[off].astype(np.int32)
    h=co.data[off]
    prod=s[r]*s[c]
    good=prod<0
    idx=np.arange(D,dtype=np.int32)
    Hsf=sp.coo_matrix((
        np.concatenate([-h[good],diag]),
        (np.concatenate([r[good],idx]),np.concatenate([c[good],idx]))
    ),shape=H.shape).tocsr()
    R=sp.coo_matrix((h[~good],(r[~good],c[~good])),shape=H.shape).tocsr()
    Ht=Hsf+R
    Habs=Hsf-R
    return Hsf,R,Ht,Habs,int(np.count_nonzero(good)),int(np.count_nonzero(~good))

def sector_vectors(Hsf,R,e,tau):
    Z=sp.csr_matrix(Hsf.shape)
    A=-Hsf
    B=-R
    M=sp.bmat([[A,Z,Z],[B,A,Z],[Z,B,A]],format="csr")
    init=np.concatenate([e,np.zeros_like(e),np.zeros_like(e)])
    out=np.asarray(sla.expm_multiply(tau*M,init),float)
    return out[:D],out[D:2*D],out[2*D:]

def one_tau(Hsf,R,Ht,Habs,e,tau):
    exact=np.asarray(sla.expm_multiply(-tau*Ht,e),float)
    absolute=np.asarray(sla.expm_multiply(-tau*Habs,e),float)
    v0,v1,v2=sector_vectors(Hsf,R,e,tau)
    q0=float(np.sum(v0))
    q1=float(-np.sum(v1))
    q2=float(np.sum(v2))
    qabs=float(np.sum(absolute))
    signed=float(np.sum(exact))
    q012=q0+q1+q2
    trunc_signed=q0-q1+q2
    return {
        "tau":float(tau),
        "q0":q0,"q1":q1,"q2":q2,
        "p0":q0/qabs,"p1":q1/qabs,"p2":q2/qabs,
        "tail_ge3":max(0.0,1.0-q012/qabs),
        "mean_m_le2_over_captured":(q1+2*q2)/q012,
        "residual_average_sign_exact":signed/qabs,
        "residual_average_sign_mle2":trunc_signed/q012,
        "signed_mass_capture":trunc_signed/signed if signed!=0 else None,
        "sector_sign_checks":[float(v0.min()),float(v1.max()),float(v2.min())],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ncols",type=int,default=1)
    ap.add_argument("--times",default="0.05,0.1,0.25")
    ap.add_argument("--seed",type=int,default=20260930)
