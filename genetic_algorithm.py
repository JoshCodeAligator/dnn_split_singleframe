import numpy as np

# --- real-world data from your trace & model ---
outputs = np.load("outputVehNum_all.npy")   # shape (total_slots, rsu*zones)
vcl     = np.array([0, 18e9, 33.55e9, 54.6172e9, 67.2072e9, 89.6351e9, 119.2351e9])
rcl     = np.array([119.2351e9, 101.2351e9, 85.6851e9, 64.6179e9, 52.0279e9, 29.6e9, 0])
Sm_k    = (np.array([32.8e6,16.4e6,8.2e6,4.1e6,2.05e6,2.05e6,1.02e6]) * 8)

# System constants
Fv   = 200e9     # Vehicle FLOPS
FRm  = 1e12      # RSU FLOPS
BR   = 20e6      # Bandwidth Hz
Pt_v = 0.2       # Tx power W
G    = 1.0       # gain
η    = 2.7       # path‐loss exponent
σ2   = 8e-14     # noise W

# dummy distances (m) for each RSU×zone
zones_distances = np.full((3,5), 10.0)

def collected_data():
    return {
        # Eq.7 params
        "Dmax":       0.2,
        "eps":        1e-3,
        "b":          5.0,
        # GA params
        "pop_size":   200,
        "gens":       500,
        "tourn_size": 3,
        "p_crossover":0.9,
        "p_mutation": 0.2,
        "stagnation": 50,
        # geometry
        "rsu":        3,
        "zones":      5,
        "n_splits":   len(vcl)
    }

cfg = collected_data()

def evaluate_population(pop_n, M, dist, alpha, beta):
    """
    Eq.7: sum_k M[k] * [1 - exp(-b*( alpha[k]+beta[k] + slack[k] ))]
      slack[k] = Dmax - (A_k+B_k+C_k) + eps
    pop_n: (Np,Z), M: (Z,), dist: (Z,), alpha,beta: (Z,)
    Returns: fitness (Np,)
    """
    Np, Z = pop_n.shape

    # 1) A_k = vcl[n-1]/Fv
    A = vcl[pop_n-1] / Fv                  # (Np,Z)

    # 2) B_k = M[k]*rcl[n-1]/FRm
    B = (M[None,:] * rcl[pop_n-1]) / FRm   # (Np,Z)

    # 3) C_k = M[k]*Sm_k[n-1] / rate_k
    snr  = (Pt_v*G) * dist[None,:]**(-η) / σ2
    rate = BR * np.log2(1 + snr)           # (Np,Z)
    C    = (M[None,:] * Sm_k[pop_n-1]) / rate

    slack = cfg["Dmax"] - (A+B+C) + cfg["eps"]
    slack = np.maximum(slack, 1e-12)

    X = alpha[None,:] + beta[None,:] + slack   # (Np,Z)
    return np.sum( M[None,:] * (1 - np.exp(-cfg["b"]*X)), axis=1 )

def genetic_for_rsu(M, dist, alpha, beta, seed=0):
    """
    GA to find best n ∈{1..n_splits}^Z maximizing Eq.7 (with fixed alpha,beta).
    Returns best n*, fitness.
    """
    rng = np.random.default_rng(seed)
    Z   = cfg["zones"]
    K   = cfg["n_splits"]
    Np  = cfg["pop_size"]

    # init random n‐population
    pop_n = rng.integers(1, K+1, size=(Np,Z))

    fitness = evaluate_population(pop_n, M, dist, alpha, beta)
    best_idx, best_fit = int(fitness.argmax()), float(fitness.max())
    no_imp = 0

    for gen in range(cfg["gens"]):
        # --- select + Crossover + Mutate -> new pop_n ---
        new = np.empty_like(pop_n)
        for i in range(0, Np, 2):
            # tournament pick two parents
            sel = rng.choice(Np, cfg["tourn_size"], replace=False)
            p1  = sel[int(fitness[sel].argmax())]
            sel = rng.choice(Np, cfg["tourn_size"], replace=False)
            p2  = sel[int(fitness[sel].argmax())]

            n1, n2 = pop_n[p1].copy(), pop_n[p2].copy()

            # one‐point crossover
            if rng.random() < cfg["p_crossover"]:
                pt = rng.integers(1, Z)
                n1[:pt], n2[:pt] = n2[:pt].copy(), n1[:pt].copy()

            # mutation
            for child in (n1,n2):
                for z in range(Z):
                    if rng.random() < cfg["p_mutation"]:
                        child[z] = rng.integers(1, K+1)

            new[i]   = n1
            if i+1 < Np:
                new[i+1] = n2

        pop_n = new
        fitness = evaluate_population(pop_n, M, dist, alpha, beta)

        # track best & early stop
        idx, fit = int(fitness.argmax()), float(fitness.max())
        if fit > best_fit:
            best_fit, best_idx, no_imp = fit, idx, 0
        else:
            no_imp += 1
            if no_imp >= cfg["stagnation"]:
                break

    return pop_n[best_idx], best_fit

def optimize_time_slot(t, alpha, beta):
    """
    Run GA for each RSU at timeslot t, given fixed alpha, beta.
    """
    M        = outputs[t].reshape(cfg["rsu"], cfg["zones"])
    results  = []
    for m in range(cfg["rsu"]):
        dist = zones_distances[m]
        n_star, fit = genetic_for_rsu(M[m], dist, alpha, beta, seed=42+m)
        results.append((n_star, fit))
    return results

if __name__ == "__main__":
    slot = 900

    alpha = np.array([0.1,0.2,0.2,0.2,0.1])
    beta  = np.array([0.1,0.1,0.1,0.1,0.1])
    print(f"\n=== GA‐only splits for timeslot {slot} ===\n")

    veh = outputs[slot].reshape(cfg["rsu"], cfg["zones"])
    for i,r in enumerate(veh,1):
        print(f"RSU{i} counts:", ' '.join(f"{int(x):3d}" for x in r))
    print()

    for i,(n_star,fit) in enumerate(optimize_time_slot(slot,alpha,beta),1):
        print(f"RSU{i}  best n* = {n_star},  fitness = {fit:.3f}")
