from dnn_split_singleframe import dnn_partition
from copy import deepcopy
import pandas as pd

# INPUTS
def get_inputs():
    return {
        # SCALARS
        "Fv": 200e9,     # 200 GFLOPS (higher onboard compute to reduce vehicle delay A)
        "BR": 1e9,       # 1 Gbps uplink bandwidth (reduces upload time C)
        "Dmax": 0.8,     # 800 ms max delay (relaxed deadline for feasibility margin)
        "Pt_v": 0.5,     # 500 mW transmit power 
        "G": 1.0,        # Unit gain (omnidirectional antenna)
        "η": 2.0,        # Path loss exponent (urban line-of-sight range)
        "σ2": 1e-9,      # Thermal noise power (−90 dBm ≈ 1e-9 W at room temperature)

        "FRm": [400e9, 400e9],  # 400 GFLOPS per RSU 

        # Real-World Compute Loads for Vehicle and RSU
        "vehicle_compute_load": [0, 18e9, 33.55e9, 54.6172e9, 67.2072e9, 89.6351e9, 119.2351e9],
        "rsu_compute_load":     [119.2351e9, 101.2351e9, 85.6851e9, 64.6179e9, 52.0279e9, 29.6e9, 0],

        # Feature Sizes (Sm_k):
        "Sm_k": [
            [0.09 * 8e5, 0.09 * 8e5, 0.09 * 8e5],  # RSU 1
            [0.09 * 8e5, 0.09 * 8e5]               # RSU 2
        ],

        # Distances (d_mk): The closer the distance, the better the SNR and throughput (C)
        "d_mk": [
            [5, 6, 6],   # RSU 1
            [6, 5]       # RSU 2
        ],

        # vehicles counts per zone for each RSU (Mm_k)
        "Mm_k": [
            [1, 2, 1],   # RSU 1
            [2, 1]       # RSU 2
        ]
    }
# OUTPUTS
# ----------
# For each RSU and its zones:
# • nm_k[M][K]: Optimal DNN partition index (n) selected per zone from the candidate splits.
# • αm_k[M][K], βm_k[M][K]: Fraction of RSU compute (alpha) and bandwidth (beta) allocated per zone.
# 
# Returned as a list of Pandas DataFrames — one per RSU — where each row corresponds to a zone with:
#     - zone       → zone index (1-based)
#     - n          → selected DNN split index
#     - value      → number of vehicles in the zone
#     - alpha*     → optimal α computed using Lagrangian method
#     - beta*      → optimal β computed using Lagrangian method
#     - total_delay → total delay for the zone's offloading computation


# Vehicle Mobility helper function – applies mobility to update vehicle counts per zone in each RSU.
def apply_vm(Mm_k, mobility_ratio=0.2):
    """
    Updates vehicle counts due to zone-to-zone and RSU-to-RSU mobility.
    """
    Mm_k = deepcopy(Mm_k)
    M_updated = [zone_counts[:] for zone_counts in Mm_k]  
    
    for m in range(len(Mm_k)):
        for k in range(len(Mm_k[m])):
            vehicle_count = Mm_k[m][k]
            if vehicle_count <= 0:
                continue

            moved_out = max(1, int(vehicle_count * mobility_ratio)) if vehicle_count > 0 else 0
            M_updated[m][k] -= moved_out

            # Vehicles move to next zone or previous RSU
            # Attempt to move to next zone
            if k + 1 < len(Mm_k[m]):
                M_updated[m][k + 1] += moved_out

            # Otherwise, try the previous RSU's same zone
            elif m > 0 and k < len(Mm_k[m - 1]):
                M_updated[m - 1][k] += moved_out

            # Fallback Logic
            # Restores moved vehicles back to its original zone
            else:
                M_updated[m][k] += moved_out


    return M_updated


# DNN Optimization Runner. Executes the DNN Partition function for a set number of iterations
def dnn_per_slot(inputs: dict, iterations=3, save_csv=False, mobility_ratio=0.2):
    time_slot = inputs["Dmax"]
    
    for t in range(iterations):
        current_time = round(t * time_slot, 3)
        print(f"\n===== Time Slot {t + 1} | Time: {current_time:.3f}s =====")

        # Update vehicle counts using the mobility model.

        inputs["Mm_k"] = apply_vm(inputs["Mm_k"], mobility_ratio)
        
        results = dnn_partition(
            inputs["Mm_k"], inputs["Sm_k"], inputs["d_mk"], inputs["FRm"],
            inputs["vehicle_compute_load"], inputs["rsu_compute_load"], inputs["Dmax"],
            inputs["Fv"], inputs["BR"], inputs["Pt_v"], inputs["G"], inputs["η"], inputs["σ2"]
        )

        for idx, df in enumerate(results, start=1):
            print(f"\n--- RSU {idx} ---")
            print(f"Total Vehicles: {df['value'].sum()}")
            print(df.to_string(index=False))

            if save_csv:
                df.to_csv(f"rsu_{idx}_slot_{t+1}.csv", index=False)

# main function that runs DNN partition
if __name__ == "__main__":
    inputs = get_inputs()
    dnn_per_slot(inputs, iterations=3, save_csv=False)
