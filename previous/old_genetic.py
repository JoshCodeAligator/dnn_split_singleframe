import numpy as np

# --- real-world data from your trace & model ---
outputs = np.load("outputVehNum_all.npy")   # shape (total_slots, rsu*zones)
vcl     = np.array([0, 18e9, 33.55e9, 54.6172e9, 67.2072e9, 89.6351e9, 119.2351e9])
rcl     = np.array([119.2351e9, 101.2351e9, 85.6851e9, 64.6179e9, 52.0279e9, 29.6e9, 0])  # unused in fitness
Sm_k    = np.array([32.8e6, 16.4e6, 8.2e6,  4.1e6,  2.05e6, 2.05e6, 1.02e6])            # unused in fitness

Fv      = 200e9   # vehicle compute power (GFLOPS)

def collected_data():
    return {
        # system parameters
        "Dmax":       0.2,   # seconds
        "eps":        1e-3,  # small constant in Eq.7
        "b":          5.0,   # steepness in Eq.7

        # GA hyper-parameters
        "pop_size":   200,
        "gens":       500,
        "tourn_size": 3,
        "p_crossover":0.9,
        "p_mutation": 0.2,
        "sigma":      0.02,
        "stagnation": 50,    # early stop if no improvement

        # geometry
        "rsu":        3,
        "zones":      5,
        "n_splits":   len(vcl)
    }

cfg = collected_data()

# Vectorized fitness evaluation for an entire population
def evaluate_population(pop_n, pop_a, M_zone):
    # pop_n: (pop_size, zones) integer in [1..n_splits]
    # pop_a: (pop_size, zones) float alphas summing to 1
    # M_zone: (zones,) vehicle counts
    # Compute A_k = vcl[n-1] / Fv for all individuals
    A = vcl[pop_n - 1] / Fv                       # (pop_size, zones)
    X = pop_a + (cfg["Dmax"] - A) + cfg["eps"]  # (pop_size, zones)
    return np.sum(M_zone * (1 - np.exp(-cfg["b"] * X)), axis=1)

# Genetic algorithm for one RSU
def genetic_for_rsu(M_zone, seed=None):
    rng = np.random.default_rng(seed)
    Z  = cfg["zones"]
    S  = cfg["n_splits"]
    Np = cfg["pop_size"]

    # Initialize: random splits and Dirichlet alphas
    pop_n = rng.integers(1, S+1, size=(Np, Z))
    pop_a = rng.dirichlet(np.ones(Z), size=Np)

    best_fit, no_imp = -np.inf, 0
    for gen in range(cfg["gens"]):
        fitness = evaluate_population(pop_n, pop_a, M_zone)
        top = fitness.max()
        if top > best_fit:
            best_fit, no_imp = top, 0
        else:
            no_imp += 1
            if no_imp >= cfg["stagnation"]:
                break

        new_n = np.empty_like(pop_n)
        new_a = np.empty_like(pop_a)
        idx = 0
        while idx < Np:
            # Tournament selection
            sel = rng.choice(Np, cfg["tourn_size"], replace=False)
            p1  = sel[np.argmax(fitness[sel])]
            sel = rng.choice(Np, cfg["tourn_size"], replace=False)
            p2  = sel[np.argmax(fitness[sel])]

            n1, n2 = pop_n[p1].copy(), pop_n[p2].copy()
            a1, a2 = pop_a[p1].copy(), pop_a[p2].copy()
            
            # Crossover
            if rng.random() < cfg["p_crossover"]:
                pt = rng.integers(1, Z)
                n1[:pt], n2[:pt] = pop_n[p2][:pt], pop_n[p1][:pt]
                a1[:pt], a2[:pt] = pop_a[p2][:pt], pop_a[p1][:pt]

            # Mutation on n (random reassign)
            for child_n in (n1, n2):
                if rng.random() < cfg["p_mutation"]:
                    child_n[rng.integers(Z)] = rng.integers(1, S+1)
            # Mutation on alpha
            for child_a in (a1, a2):
                if rng.random() < cfg["p_mutation"]:
                    child_a += rng.standard_normal(Z) * cfg["sigma"]
                np.clip(child_a, 1e-12, None, out=child_a)
                child_a /= child_a.sum()

            new_n[idx], new_a[idx] = n1, a1
            if idx+1 < Np:
                new_n[idx+1], new_a[idx+1] = n2, a2
            idx += 2

        pop_n, pop_a = new_n, new_a

    # Final best
    fitness = evaluate_population(pop_n, pop_a, M_zone)
    best = np.argmax(fitness)
    return pop_n[best], pop_a[best], fitness[best]

# Run GA for each RSU in a given time slot
def optimize_time_slot(t, seed=0):
    M = outputs[t].reshape(cfg["rsu"], cfg["zones"])
    results = []
    for m in range(cfg["rsu"]):
        n_opt, a_opt, fit = genetic_for_rsu(M[m], seed + m)
        results.append((n_opt, a_opt, fit))
    return results


if __name__ == "__main__":
    slot = 1002
    print(f"\n=== Optimization for Time Slot {slot} ===\n")
    veh = outputs[slot].reshape(cfg["rsu"], cfg["zones"])
    for i,row in enumerate(veh,1):
        print(f"RSU {i} vehicles: ", "  ".join(str(int(v)) for v in row))
    print()
    for i,(n_opt,a_opt,fit) in enumerate(optimize_time_slot(slot, seed=42),1):
        print(f"RSU {i} | fitness={fit:.3f}")
        print("  n splits:", ", ".join(map(str,n_opt)))
        print("  α alloc :", ", ".join(f"{x:.4f}" for x in a_opt))
        print("-"*40)
