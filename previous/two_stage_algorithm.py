import math
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD

###TWO STAGE ALGORITHM PIPELINE
def dnn_partition(Mm_k_t, Sm_k_bits, dm_k_t, FRm, vcl, rcl,
                  Dmax, Fv, BR, Pt_v, G, η, σ2):
    """
    Two-stage optimization for DNN partitioning and resource allocation.
    Stage 2: Discrete selection of split points via grouped knapsack.
    Stage 1: Continuous Lagrangian allocation of alpha, beta.
    Returns: list of DataFrames (one per RSU) with columns
    [zone, n, value, alpha*, beta*, total_delay].
    """

    # --- Stage 2: Grouped Knapsack to pick split points ---
    selections = []
    M = len(Mm_k_t)
    for m in range(M):                          # Iterates through each RSU
        # Build candidate items per zone
        zone_cands = [] 
        for k, count in enumerate(Mm_k_t[m]):   # Iterates through each zone of current RSU. 
            items = []
            for n in range(len(vcl)):           # Iterates through the 7 DNN partition points for current zone. Ensures within system constraints.
                A = vcl[n] / Fv                 # delay term A
                B = (count * rcl[n]) / FRm      # delay term B
                d = dm_k_t[m][k]                # average distance of vehicles in current zone
                S_bits = Sm_k_bits[n]           # intermediate data size based on DNN partion point (n). n will be used as an index value to extract the appropriate data size from Sm_k_bits.
                snr = (Pt_v * G * d**(-η)) / σ2 
                T = BR * math.log2(1 + snr)     # computes uplink throughput based on snr and total available RSU bandwidth.
                C = (count * S_bits) / T        # delay term C
                E = Dmax - A                    # remaining delay budget
                
                #Filters out DNN partitions that do not meet system constraints for each zone.
                # The zone candidates list stores rsu, zone, and vehicle count information for valid partition points.
                if E <= 0:
                    continue
                s = math.sqrt(B*C)
                α = (B + s) / E
                β = (C + s) / E
                if not (0 < α <= 1 and 0 < β <= 1):
                    continue
                Dtotal = A + B/α + C/β
                if Dtotal <= Dmax:
                    items.append({
                        "m": m, "zone": k, "n": n+1, "value": count,
                        "α": α, "β": β,
                        "weight": max(min(α, β), α + β - 1)
                    })
            if not items:
                items.append({
                    "m":      m,
                    "zone":   k,
                    "n":      len(vcl),
                    "value":  count,
                    "weight": 0.0
                })
            zone_cands.append(items)
        # Using Integer Linear Problem for Group Knapsack Solution.
        # Goal: To find the best combination of DNN partitions across all zones in RSU.

        # 1) set up problem
        prob = LpProblem(sense=LpMaximize)
        vars_ = []
        for k, group in enumerate(zone_cands):
            for i, item in enumerate(group):
                v = LpVariable(f"x_{k}_{i}", cat=LpBinary)
                vars_.append((k, item, v))

        # 2) objective: to maximize total vehicles served
        prob += lpSum(item["value"] * v for k, item, v in vars_)

        # 3) exactly one split per zone
        for k in range(len(zone_cands)):
            prob += lpSum(v for k0, item, v in vars_ if k0 == k) == 1

        # 4) single resource constraint: sum of item weights ≤ 1
        prob += lpSum(item["weight"] * v for _, item, v in vars_) <= 1

        # 5) solve
        prob.solve(PULP_CBC_CMD(msg=False))

        # 6) collect selections
        sel = [item for k, item, v in vars_ if v.value() == 1]
        selections.append(sel)

    # --- Stage 1: Closed-Form Lagrangian allocation ---
    # Uses Convex Optimization techniques to find optimal alpha and beta values. 
    # alpha - fractional share of RSU Compute Power. beta - fractional share of RSU bandwidth. both must be <= 1.
    results = []
    for sel in selections:
        rows = []
        for item in sel:
            m     = item["m"]
            k     = item["zone"]
            n_idx = item["n"] - 1
            count = item["value"]

            # Local Only (LO): No offloading to RSU. Vehicle runs entire DNN.
            if item["n"] == len(vcl):
                A = vcl[n_idx] / Fv
                alpha_star = 0.0
                beta_star  = 0.0
                Dtotal     = A
            else:
                # normal collaborative case: Vehicles offload tasks to RSU.
                A = vcl[n_idx] / Fv
                B = (count * rcl[n_idx]) / FRm
                d = dm_k_t[m][k]
                S = Sm_k_bits[n_idx]
                snr = (Pt_v * G * d**(-η)) / σ2
                T   = BR * math.log2(1 + snr)
                C   = (count * S) / T
                E   = Dmax - A

                # Even if local alone breaks the deadline, treat it as a local only (LO) case.
                if E <= 0:
                    alpha_star = 0.0
                    beta_star  = 0.0
                    Dtotal     = A
                else:
                    s = math.sqrt(B * C)
                    alpha_star = (B + s) / E
                    beta_star  = (C + s) / E
                    Dtotal     = A + (B/alpha_star) + (C/beta_star)

            rows.append({
                "zone":        k+1,
                "n":           n_idx+1,
                "value":       count,
                "alpha*":      alpha_star,
                "beta*":       beta_star,
                "total_delay": Dtotal
            })

        # Orders each RSU's results based on zone index.
        df = pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)
        results.append(df)

    return results