import math
import itertools
import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, lpSum, LpBinary, PULP_CBC_CMD

###INPUTS
#SCALARS
Fv = 1e9        # 1 GFLOPS (realistic onboard compute for an edge vehicle or embedded system)
FRm = [1e9]     # 1 GFLOPS for RSU 
BR = 100e6      # 100 Mbps uplink bandwidth (typical for 4G LTE/5G shared uplink capacity)
Dmax = 0.04     # 40 ms max delay (standard for real-time safety or perception tasks)
Pt_v = 0.2      # 200 mW transmit power (realistic for V2X communication)
G = 1.0         # Unit gain (omnidirectional antenna)
η = 2.0         # Path loss exponent (urban line-of-sight range)
σ2 = 1e-9       # Thermal noise power (−90 dBm ≈ 1e-9 W at room temperature)

# Total compute cost for vehicle-side (GFLOPs). 
vehicle_compute_load = [0, 18, 33.55, 54.6172, 67.2072, 89.6351, 119.2351]  # size [7] - DNN has 7 layers (N=7)

# Total compute cost for RSU-side (GFLOPs)
rsu_compute_load = [119.2351, 101.2351, 85.6851, 64.6179, 52.0279, 29.6, 0]  # size [7]

#Zone-Specific Values
Sm_k = [[0.5 * 8e6, 0.8 * 8e6, 0.6 * 8e6]]  # bits per vehicle. 0.5–0.8 MB common for intermediate features
d_mk = [[30, 40, 50]]                       # meters. RSUs in urban corridors often spaced ~50m

#THE ONLY DYNAMIC INPUT (TESTING)
Mm_k = [[3, 5, 4]]  # 1 RSU with 3 zones

#OUTPUTS 
'''
nm_k[M][K] - ideal combination of DNN partition points for all zones in RSU. (Found in Step 2)
αm_k[M][K], βm_k[M][K] - fraction of RSU compute power and bandwidth allocated per zone. (Found in Step 3)
'''

###3-STEP ALGORITHM PIPELINE
#Step 1 - DNN Split Function that returns valid DNN splits per zone.
valid_nm_k_2D = []

def n_filter(Mm_k):
    result = []  # one entry per zone

    for zone, vehicle_count in enumerate(Mm_k[0]):  # M=1 RSU assumed
        candidates = []

        #Computing Alpha and Beta and Checking if it is within System Constraints
        for n in range(0, 7):  # DNN split options
            A = vehicle_compute_load[n] / Fv
            B = (vehicle_count * rsu_compute_load[n]) / FRm[0]

            d = d_mk[0][zone]
            Sm = Sm_k[0][zone]
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
                    weight = max(min(alpha, beta), alpha + beta - 1)
                    candidates.append({
                        "zone": zone,
                        "dnn_split": n + 1,
                        "value": vehicle_count,
                        "weight": weight,
                        "avg_weight": vehicle_count * weight,
                        "alpha": alpha,
                        "beta": beta
                    })
        result.append(candidates)

    return result

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
    prob.solve(PULP_CBC_CMD(msg=0))

    if LpStatus[prob.status] != "Optimal":
        print(f"Warning: Problem not solved optimally. Could not include all zones. Status = {LpStatus[prob.status]}\n")

    # Extract the selected items
    selected_items = selected_items = [
        {"zone": item["zone"], "dnn_split": item["dnn_split"], "value": item["value"]}
        for _, _, item, var in variables if var.varValue == 1
    ]

    return selected_items

#Step 3 - Optimizing and Finalizing Alpha and Beta values. 
# The Lagrangian Method will be used on the selected set of dnn splits from Step 2.
def optimal_alpha_beta(best_selection):
    finalized = []
    
    for entry in best_selection:
        k = entry['zone']
        n = entry['dnn_split'] - 1  
        value = entry['value']

        #Closed-Form Lagrangian Method Calculations
        A = vehicle_compute_load[n] / Fv
        B = (value * rsu_compute_load[n]) / FRm[0]
        d = d_mk[0][k]
        Sm = Sm_k[0][k]
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
            "zone": k,
            "n": n + 1,
            "value": value,
            "alpha*": optimal_alpha,
            "beta*": optimal_beta,
            "total_delay": A + (B / optimal_alpha) + (C / optimal_beta)
        })

    return finalized

#Testing Output
###n_filter algorithm - Step 1
valid_nm_k_2D= n_filter(Mm_k)


###group knapsack for discrete optimization - Step 2
#RSU can NOT offload vehicle workload
best_selection = group_knapsack(valid_nm_k_2D)
if not best_selection:
    print("No valid selection found.")

#RSU can offload vehicle workload
else:
    #Calculates Total Vehicle Count
    total_vehicles = sum(item["value"] for item in best_selection)

    ###finalized alpha* and beta* - Step 3
    final_output = optimal_alpha_beta(best_selection)
    
    # Tabulates Values in a Pandas Dataframe according to Zone Order
    df_output = pd.DataFrame(final_output).sort_values("zone")   
    print(f"Total Vehicle Count: {total_vehicles}\n")
    print(df_output)


