from dnn_split_singleframe import dnn_partition
import pandas as pd

# INPUTS
def get_inputs():
    return {
        "Fv": 5e9,
        "BR": 100e6,
        "Dmax": 0.2,
        "Pt_v": 0.5,
        "G": 1.0,
        "η": 2.0,
        "σ2": 1e-9,
        "FRm": [10e9, 10e9],
        "vehicle_compute_load": [0, 0.1e9, 0.2e9, 0.3e9, 0.4e9, 0.5e9, 0.6e9],
        "rsu_compute_load":     [0.6e9, 0.5e9, 0.4e9, 0.3e9, 0.2e9, 0.1e9, 0],
        "Sm_k": [
            [0.1 * 8e5, 0.1 * 8e5, 0.1 * 8e5],
            [0.1 * 8e5, 0.1 * 8e5]
        ],
        "d_mk": [
            [10, 15, 20],
            [10, 15]
        ],
        "Mm_k": [
            [1, 2, 1],
            [2, 1]
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
