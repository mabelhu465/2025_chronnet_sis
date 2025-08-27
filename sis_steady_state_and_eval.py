import numpy as np
from scipy import optimize

def nimfa_sis_steady_state_root_1(W, beta, gamma):
    """
    solve the meta-stable solution of the continuous-time N-Intertwined SIS model
    by solving R(P)=0 with scipy.optimize.root, now providing jac.
    """
    import numpy as np
    from scipy import optimize
    import warnings

    n = W.shape[0]
    p0 = np.full(n, 1.0)

    def eq(P):
        WP = W.dot(P)
        return beta*(1 - P) * WP - gamma*P


    def jac(P):
        WP = W.dot(P)
        # part 1：β * diag(1-P) @ W
        # part 2：-β * diag(WP)
        # part 3: γ*I
        return beta * ((1 - P)[:, None] * W - np.diag(WP)) - gamma * np.eye(n)

    # print the initial guess
    #r0 = eq(p0)
    #print("||f(p0)|| =", np.linalg.norm(r0), "max|f(p0)| =", np.max(np.abs(r0)))
 
    try:
        sol = optimize.root(
            eq,
            p0,
            jac=jac,              
            method='hybr',
            tol=1e-8,
            options={'xtol':1e-8, 'maxfev':20000}
        )
        print("Solver status:", sol.success, sol.status, sol.message,
              "func evals:", sol.nfev, "iterations:", getattr(sol, 'nit', None))
        print("  sol.x stats: min =", sol.x.min(),
              "max =", sol.x.max(), "mean =", sol.x.mean())
        print("  ||f(sol.x)|| =", np.linalg.norm(eq(sol.x)))

        if not sol.success:
            raise ValueError(f"Fail to solve SIS steady state: {sol.message}")
        v_sis = sol.x

    except ValueError:
        v_sis = np.full(n, 0.0)
        warnings.warn(f"τ={beta:.3f} not converged, set to 0")

    return v_sis



def eq(v,W,beta,gamma):
        # R_i = beta*(1-P_i)*(W P)_i - gamma*P_i
        WP = W.dot(v)
        return beta*(1 - v) * WP - gamma*v




def recognition_quality(v, phi):
    """
    v: prediction scores, shape=(N,)
    phi: true labels or ability values, shape=(N,)
    Returns: AUC-style recognition quality: (1/N)*sum_{k=1..N} recall@k
    """
    N = len(v)
   
    order_v   = np.argsort(-v)
    order_phi = np.argsort(-phi)

    overlaps = []
    top_v_set   = set()
    top_phi_set = set()

    for k in range(1, N+1):
        top_v_set.add(order_v[k-1])
        top_phi_set.add(order_phi[k-1])
        overlaps.append(len(top_v_set & top_phi_set) / k)

    return float(np.mean(overlaps))


def jsd_from_samples(P, Q, bins=100,scale ='minmax'):
    ## Jensen-Shannon divergence
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    
    if scale == 'minmax':
        P = (P - np.min(P)) / (np.max(P) - np.min(P))
        Q = (Q - np.min(Q)) / (np.max(Q) - np.min(Q))
    else:
        pass

    lo = min(np.min(P), np.min(Q))
    hi = max(np.max(P), np.max(Q))
    bin_edges = np.linspace(lo, hi, bins + 1)
    P_cnt, _ = np.histogram(P, bins=bin_edges, density=False)
    Q_cnt, _ = np.histogram(Q, bins=bin_edges, density=False)

    P_prob = P_cnt / np.sum(P_cnt)
    Q_prob = Q_cnt / np.sum(Q_cnt)

    M = 0.5 * (P_prob + Q_prob)
    def H(X):
        Xp= X[X > 0]
        return -np.sum(Xp * np.log2(Xp))

    return H(M) - 0.5*(H(P_prob) + H(Q_prob))


