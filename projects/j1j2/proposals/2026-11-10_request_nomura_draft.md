Draft request to Yusuke Nomura (for the PI to send; nothing has been sent)

Subject: Request: optimized RBM+PP parameters for the 8x8 J1-J2 model (PRX 11, 031034 / arXiv:2005.14142)

Dear Dr. Nomura,

we work on fixed-node lattice Monte Carlo with neural-network guide wavefunctions for the J1-J2 Heisenberg model, and we use
pair-product (RBM+PP) states as guides. For a fair benchmark at L = 8 we would like to start from your optimized state rather than
from our own, which is about 2e-4 per site above your published value.

Would you be willing to share, for the 8x8 lattice, periodic boundaries, J2/J1 = 0.5 (Table I of arXiv:2005.14142, E/N = -0.498886(1)):
1. the optimized variational parameters: the pair amplitudes f_ij (with any sublattice or symmetry structure imposed) and the RBM
   parameters (visible/hidden biases and couplings, number of hidden units, complex or real), e.g. the mVMC output files (zqp_opt.dat
   or the equivalent parameter files) together with the input files that define their meaning (orbital index map, RBM definition,
   projection settings);
2. the quantum-number projection used for that number (spatial group elements, spin parity, total-spin projection on or off), and
   the sign convention of the wavefunction (e.g. Marshall sign or none);
3. if available, the optimization settings (SR shift, step size, samples per step, number of steps) and the final energy with its
   statistical error for that parameter set.

We would cite your work for any result using these parameters and are happy to share our results with you.

With best regards,
[PI name, affiliation]
