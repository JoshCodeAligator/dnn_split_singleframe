import math
import itertools
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, lpSum, LpBinary, PULP_CBC_CMD

###3-STEP ALGORITHM PIPELINE
def dnn_partition(Mm_k, Sm_k, d_mk, FRm, vehicle_compute_load, rsu_compute_load, Dmax,
                  Fv, BR, Pt_v, G, η, σ2):
    #Step 1 - DNN Split Function that returns valid DNN splits per zone.
    valid_nm_k = []

    def n_filter(Mm_k):
        results = []
        for m in range(len(Mm_k)):
            rsu_results = []  
            for zone, vehicle_count in enumerate(Mm_k[m]):
                candidates = []

                #Computing Alpha and Beta and Checking if it is within System Constraints
                #Refer to revised document to make changes with calculations.
                #Fix throughput calculation
                for n in range(0, 7):
                    A = vehicle_compute_load[n] / Fv
                    B = (vehicle_count * rsu_compute_load[n]) / FRm[m]
                    d = d_mk[m][zone]
                    Sm = Sm_k[m][zone]
                    snr = (Pt_v * G * d ** (-η)) / σ2
                    uplink_throughput = BR * math.log2(1 + snr)
                    C = (vehicle_count * Sm) / uplink_throughput
                    E = Dmax - A
                    
                    if E <= 0:
                        continue
                    sqrt_BC = math.sqrt(B * C)
                    alpha = (B + sqrt_BC) / E
                    beta = (C + sqrt_BC) / E
                    if 0 < alpha <= 1 and 0 < beta <= 1:
                        total_delay = A + (B / alpha) + (C / beta)
                        if total_delay <= Dmax:
                            candidates.append({
                                "rsu": m,
                                "zone": zone,
                                "dnn_split": n + 1,
                                "value": vehicle_count,
                                "weight": max(min(alpha, beta), alpha + beta - 1),
                                "alpha": alpha,
                                "beta": beta
                            })
                rsu_results.append(candidates)
            results.append(rsu_results)
        return results

    #Step 2 - Group Knapsack Problem - Discrete Optimization + Convex Optimization. 
    #Selects the best combination of dnn partition points across all zones.
    def group_knapsack(zone_candidates):
        prob = LpProblem("GroupKnapsack", LpMaximize)
        variables = []

        # Create binary decision variables
        for group_idx, group in enumerate(zone_candidates):
            for item_idx, item in enumerate(group):
                var = LpVariable(f"x_{group_idx}_{item_idx}", cat=LpBinary)
                variables.append((group_idx, item_idx, item, var))

        # Objective: Maximize ∑ value × weight
        prob += lpSum(item['value'] * item['weight'] * var for _, _, item, var in variables)

        # Constraint: Select exactly one item per group
        for group_idx in range(len(zone_candidates)):
            prob += lpSum(var for g_i, _, _, var in variables if g_i == group_idx) == 1

        # Constraint: Total α and β usage within limits
        prob += lpSum(item['alpha'] * var for _, _, item, var in variables) <= 1
        prob += lpSum(item['beta'] * var for _, _, item, var in variables) <= 1

        # Solve the problem
        prob.solve(PULP_CBC_CMD())

        if LpStatus[prob.status] != "Optimal":
            print(f"Warning: Problem not solved optimally. Could not include all zones. Status = {LpStatus[prob.status]}\n")

        # Extract the selected items
        selected_items = [
            {"rsu": item["rsu"], "zone": item["zone"], "dnn_split": item["dnn_split"], "value": item["value"]}
            for _, _, item, var in variables if var.varValue == 1
        ]

        return selected_items

    # Step 3 - Optimizing and Finalizing Alpha and Beta values. 
    # The Lagrangian Method will be used on the selected set of dnn splits from Step 2.
    # Refer to revised document to make changes with calculations.

    #Make value clear that it is for vehicle count.
    def optimal_alpha_beta(best_selection):
        finalized = []
        
        for entry in best_selection:
            m = entry['rsu']
            k = entry['zone']
            n = entry['dnn_split'] - 1  
            value = entry['value']

            # Closed-Form Lagrangian Method Calculations
            # Partial Derivatives, set it to zero
            # Refer to Screenshots
            A = vehicle_compute_load[n] / Fv
            B = (value * rsu_compute_load[n]) / FRm[m]
            d = d_mk[m][k]
            Sm = Sm_k[m][k]
            snr = (Pt_v * G * d ** (-η)) / σ2
            uplink_throughput = BR * math.log2(1 + snr)
            C = (value * Sm) / uplink_throughput
            E = Dmax - A

            if E <= 0:
                continue

            #Re-computed alpha and beta values
            optimal_alpha = (B + math.sqrt(B * C)) / E
            optimal_beta = (C + math.sqrt(B * C)) / E

            finalized.append({
                "RSU": m,
                "zone": k + 1,
                "n": n + 1,
                "value": value,
                "alpha*": optimal_alpha,
                "beta*": optimal_beta,
                "total_delay": A + (B / optimal_alpha) + (C / optimal_beta)
            })
        return finalized

    #Testing Output
    ###n_filter algorithm - Step 1
    valid_nm_k = n_filter(Mm_k)
    dnn_results = []
    
    ###group knapsack for discrete optimization - Step 2
    #RSU can NOT offload vehicle workload
    for zone_candidates in valid_nm_k:
        best_selection = group_knapsack(zone_candidates)
        if not best_selection:
            print("No valid selection found.")

        #RSU can offload vehicle workload
        else:
            ###finalized alpha* and beta* - Step 3
            final_output = optimal_alpha_beta(best_selection)
            
            # Tabulates Values in a Pandas Dataframe according to Zone Order
            df_output = pd.DataFrame(final_output).sort_values(["RSU", "zone"]) 
            
            #Removing RSU column due to redundancy
            df_output = df_output.drop(columns=["RSU"])
            dnn_results.append(df_output)

    return dnn_results

