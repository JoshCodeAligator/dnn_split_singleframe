from dnn_split_singleframe import dnn_partition
from copy import deepcopy
import pandas as pd

# UPDATED ALL INPUTS WITH REAL WORLD INFORMATION.
# Note: BDD100K Dataset using YOLOv5x Model
def get_inputs():
    vcl = [0, 18e9, 33.55e9, 54.6172e9, 67.2072e9, 89.6351e9, 119.2351e9] # vehicle compute load for each DNN partition point.
    rcl = [119.2351e9, 101.2351e9, 85.6851e9, 64.6179e9, 52.0279e9, 29.6e9, 0] # rsu compute load for each DNN partition point.
    Sm_k = [32.8e6, 16.4e6, 8.2e6, 4.1e6, 2.05e6, 2.05e6, 1.02e6] # (bytes). Real feature map data size per DNN layer (1-7) from YOLOv5x model.
    
    return {
        "Fv": 5e12,      # Compute Power for for all vehicles - 5 TFLOPS.
        "FRm": 8.1e12,   # Compute Power for all RSUs (8.1 TFLOPS, e.g., Tesla T4 GPU)
        "BR": 20e6,      # 20 MHz LTE uplink
        "Dmax": 0.08,    # 80 ms max delay. Value must be 100 milliseconds or less. 
        "Pt_v": 0.2,     # 0.2 W = 23 dBm
        "G": 1.0,        # Unit gain (omnidirectional antenna)
        "η": 2.7,        # Path loss exponent (urban line-of-sight range)
        "σ2": 8e-14,     # Thermal noise power over 20 MHz

        # Real-World Compute Loads for Vehicle and RSU
        "vehicle_compute_load": vcl,
        "rsu_compute_load": rcl,

        # Converting feature size Sm_k from bytes to bits.
        "Sm_k_bits": [x * 8 for x in Sm_k],

        # Distances (d_mk): meters - The closer the distance, the better the SNR and throughput (C)
        # urban link distance from Hosseinalipour et al. (5–10 m)
        "dm_k": [
            [5.5, 6.2, 5.8],        # RSU 1 zones
            [6.1, 5.9, 7.0, 6.8]    # RSU 2 zones
        ]

        # average dm_k (5-10m)
        # Vehicle Count based on Mobility model (Mm_k) - (1-2)

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
# Some Real World Trace. Fix Mobility Ratio Parameter Issue.
# Make algorithm more versatile

#Implement formula excluding mobility ratio
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
