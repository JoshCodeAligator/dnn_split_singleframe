import numpy as np
import math

# Recorded vehicle counts: shape (total_slots, rsu*zones)
outputs = np.load("outputVehNum_all.npy")

# DNN partition compute loads (7 split points)
vcl  = np.array([0, 18e9, 33.55e9, 54.6172e9, 67.2072e9, 89.6351e9, 119.2351e9])
rcl  = np.array([119.2351e9, 101.2351e9, 85.6851e9, 64.6179e9, 52.0279e9, 29.6e9, 0])

# Feature‐map sizes (bytes → bits for Eq. C if needed)
Sm_k = np.array([32.8e6, 16.4e6, 8.2e6, 4.1e6, 2.05e6, 2.05e6, 1.02e6]) * 8

# System constants
Fv   = 200e9     # Vehicle FLOPS
FRm  = 1e12      # RSU FLOPS
BR   = 20e6      # Uplink bandwidth (Hz)
Pt_v = 0.2       # Transmit power (W)
G    = 1.0       # Antenna gain
η    = 2.7       # Path‐loss exponent
σ2   = 8e-14     # Noise power (W)

# Per‐RSU, per‐zone distances (m). Shape = (rsu, zones).
# we use 10m dummy:
zones_distances = np.full((3, 5), 10.0)

def collected_data():
    return {
        # Delay model
        "Dmax":       0.2,   # s
        "eps":        1e-3,  # small epsilon
        "b":          5.0,   # steepness in Eq.7

        # GA hyper‐parameters
        "pop_size":   200,
        "gens":       500,
        "tourn_size": 3,
        "p_crossover":0.9,
        "p_mutation": 0.2,
        "sigma":      0.02,
        "stagnation": 50,    # early stopping

        # Geometry
        "rsu":        3,
        "zones":      5,
        "n_splits":   len(vcl)
    }

cfg = collected_data()

def evaluate_population(pop_n, pop_a, M_zone, dist_zone):
    """
    Vectorized Eq.7 fitness for all individuals in population.
      pop_n: (Np, Z) integers in [1..n_splits]
      pop_a: (Np, Z) floats summing to 1
      M_zone: (Z,) vehicle counts
      dist_zone: (Z,) distances (m) from RSU to zones
    Returns:
      fitness: (Np,) array
    """
    # 1) Vehicle‐compute delay A_k = vcl[n-1]/Fv
    A = vcl[pop_n - 1] / Fv                           # (Np, Z)

    # 2) RSU‐compute delay B_k = (M_zone * rcl[n-1]) / FRm
    B = (M_zone[None,:] * rcl[pop_n - 1]) / FRm       # (Np, Z)

    # 3) Uplink delay C_k = (M_zone * Sm_k[n-1]) / rate
    snr = (Pt_v * G) * dist_zone[None,:]**(-η) / σ2
    rate = BR * np.log2(1 + snr)                      # (Np, Z)
    C = (M_zone[None,:] * Sm_k[pop_n - 1]) / rate     # (Np, Z)

    # Slack = Dmax − (A + B + C) + eps, clamped ≥ 1e‐12
    slack = cfg["Dmax"] - (A + B + C) + cfg["eps"]
    slack = np.maximum(slack, 1e-12)

    # X = α + slack
    X = pop_a + slack                                 # (Np, Z)

    # Eq.7: sum_k M_k [1 − exp(−b X_k)]
    return np.sum(M_zone[None,:] * (1 - np.exp(-cfg["b"] * X)), axis=1)

def genetic_for_rsu(M_zone, seed, dist_zone):
    """
    Run GA to optimize (n, α) for one RSU.
    Returns best split indices n*, allocations α*, and fitness.

    alpha is an input, not output.
    consider beta as well, not only alpha.
    to test genetic algorithm, randomly generate fitness.
    sum of alpha <=1, sum of beta <=1 

    input: alpha, beta
    output: n
    """
    rng = np.random.default_rng(seed)
    Z   = cfg["zones"]
    S   = cfg["n_splits"]
    Np  = cfg["pop_size"]

    # Initialize population
    pop_n = rng.integers(1, S+1, size=(Np, Z))        # discrete splits
    pop_a = rng.dirichlet(np.ones(Z), size=Np)        # α allocations

    best_fit, no_imp = -np.inf, 0

    for _ in range(cfg["gens"]):
        # compute fitness for all individuals
        fitness = evaluate_population(pop_n, pop_a, M_zone, dist_zone)
        top = fitness.max()

        # early stopping
        if top > best_fit:
            best_fit, no_imp = top, 0
        else:
            no_imp += 1
            if no_imp >= cfg["stagnation"]:
                break

        # create next generation
        new_n = np.empty_like(pop_n)
        new_a = np.empty_like(pop_a)
        idx = 0
        while idx < Np:
            # tournament selection
            sel = rng.choice(Np, cfg["tourn_size"], replace=False)
            p1  = sel[np.argmax(fitness[sel])]
            
            sel = rng.choice(Np, cfg["tourn_size"], replace=False)
            p2  = sel[np.argmax(fitness[sel])]

            n1, n2 = pop_n[p1].copy(), pop_n[p2].copy()
            a1, a2 = pop_a[p1].copy(), pop_a[p2].copy()

            # crossover
            if rng.random() < cfg["p_crossover"]:
                pt = rng.integers(1, Z)
                # for n
                n1[:pt], n2[:pt] = n2[:pt], n1[:pt]
                

            # mutation on n
            for child_n in (n1, n2):
                if rng.random() < cfg["p_mutation"]:
                    child_n[rng.integers(Z)] = rng.integers(1, S+1)
            

            new_n[idx], new_a[idx] = n1, a1
            if idx+1 < Np:
                new_n[idx+1], new_a[idx+1] = n2, a2
            idx += 2

        pop_n, pop_a = new_n, new_a

    # pick final best
    fitness = evaluate_population(pop_n, pop_a, M_zone, dist_zone)
    best = np.argmax(fitness)
    return pop_n[best], pop_a[best], fitness[best]

def optimize_time_slot(t, seed=0):
    M = outputs[t].reshape(cfg["rsu"], cfg["zones"])
    results = []
    for m in range(cfg["rsu"]):
        dist_zone = zones_distances[m]
        n_opt, a_opt, fit = genetic_for_rsu(M[m], seed + m, dist_zone)
        results.append((n_opt, a_opt, fit))
    return results

if __name__ == "__main__":
    slot = 1002
    
    print(f"\n=== Optimization for Time Slot {slot} ===\n")

    # vehicle counts
    veh = outputs[slot].reshape(cfg["rsu"], cfg["zones"])
    for i, row in enumerate(veh, 1):
        print(f"RSU {i} vehicles:  " + "  ".join(f"{int(v):4d}" for v in row))
    print()

    # run GA & print
    for i, (n_opt, a_opt, fit) in enumerate(optimize_time_slot(slot, seed=42), 1):
        print(f"RSU {i}  |  fitness = {fit:.3f}")
        print("  n splits:", ", ".join(str(int(x)) for x in n_opt))
        print("  α alloc :", ", ".join(f"{x:.4f}" for x in a_opt))
        print("-" * 40)
