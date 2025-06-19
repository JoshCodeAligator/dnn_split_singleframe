from dnn_split_singleframe import dnn_partition
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


# DNN Optimization Runner. Executes the DNN Partition function for a set number of iterations
def dnn_per_slot(inputs: dict, iterations=3, save_csv=False):
    time_slot = inputs["Dmax"]
    
    for t in range(iterations):
        current_time = round(t * time_slot, 3)
        print(f"\n===== Time Slot {t + 1} | Time: {current_time:.3f}s =====")

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
