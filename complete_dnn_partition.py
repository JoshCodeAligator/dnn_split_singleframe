from two_stage_algorithm import dnn_partition
from copy import deepcopy
import pandas as pd

# UPDATED ALL INPUTS WITH REAL WORLD INFORMATION.
# Note: DNN model is trained on the BDD100K Dataset using YOLOv5x Model.
# Goal: main function (last code block) to run the two stage algorithm (dnn_partition) repeatedly.

def real_world_data():
    vcl = [0, 18e9, 33.55e9, 54.6172e9, 67.2072e9, 89.6351e9, 119.2351e9] # vehicle compute load for each DNN partition point. (7 partition points)
    rcl = [119.2351e9, 101.2351e9, 85.6851e9, 64.6179e9, 52.0279e9, 29.6e9, 0] # rsu compute load for each DNN partition point. (7 partition points)
    Sm_k = [32.8e6, 16.4e6, 8.2e6, 4.1e6, 2.05e6, 2.05e6, 1.02e6] # (bytes). Real feature map data size per DNN layer (1-7) from YOLOv5x model.
    Dmax = 0.08
    
    avg_speed = 27.78  # (m/s) - average vehicle speed for 1 vehicle on highway. 100 km/h. Unused value.
    rsu_coverage = 200 # RSU covers 200 meters (m).
    
    # These are the initial vehicle counts at time slot 0.
    rsu_vehicle_counts = [
            [2, 1, 2],              # RSU 1 - 3 zones
            [2, 2, 1, 1]            # RSU 2 - 4 zones
        ]

    # For each RSU, we'll compute the length of each zone based on RSU coverage and # of zones.
    zone_lengths = [rsu_coverage / len(zones) for zones in rsu_vehicle_counts]
    
    return {
        "Fv": 5e12,      # Compute Power for for all vehicles - 5 TFLOPS.
        "FRm": 8.1e12,   # Compute Power for all RSUs (8.1 TFLOPS, e.g., Tesla T4 GPU)
        "BR": 20e6,      # 20 MHz LTE uplink
        "Dmax": Dmax,    # 80 ms max delay. Value must be 100 milliseconds or less. Same as timeslot.
        "Pt_v": 0.2,     # 0.2 W = 23 dBm
        "G": 1.0,        # Unit gain (omnidirectional antenna)
        "η": 2.7,        # Path loss exponent (urban line-of-sight range)
        "σ2": 8e-14,     # Thermal noise power over 20 MHz
        
        # Real-World Compute Loads for Vehicle and RSU
        "vehicle_compute_load": vcl,
        "rsu_compute_load": rcl,
        "zone_lengths": zone_lengths,

        # Converting feature size Sm_k from bytes to bits (Throughput calculation uses bits.)
        "Sm_k_bits": [x * 8 for x in Sm_k],

        # Vehicle Count based on Mobility model (Mm_k_t) - (1-2)
        "Mm_k_t": rsu_vehicle_counts
    }

# ASSUMPTION: Maximum of 1 vehicle may exit the RSU at a time. Computes Vehicles
def compute_Mout(Mlocal, speeds, timeslot, zone_lengths):
    Mout = []
    for m in range(len(Mlocal)):
        Mout_rsu = []
        for k in range(len(Mlocal[m])):
            local_count = Mlocal[m][k]
            speed = speeds[m][k]
            zone_length = zone_lengths[m]
            
            # Distance vehicles can travel in the slot
            travel_dist = speed * timeslot
            
            # Fraction of vehicles in the exit region of the zone
            exit_fraction = min(1.0, travel_dist / zone_length)
            
            # Number of vehicles exiting
            exiting = int(exit_fraction * local_count)
            Mout_rsu.append(min(exiting, local_count))
        Mout.append(Mout_rsu)
    return Mout


# Updates per-zone vehicle count for current time slot (Dmax). - Vehicle Mobility Model
# Mlocal = Mm,k,t, timeslot = Dmax. Based on vehicle speeds and avg vehicle distance to RSU.
def apply_vm(Mlocal, speeds, timeslot, zone_lengths):
    Mout = compute_Mout(Mlocal, speeds, timeslot, zone_lengths)
    M_updated = deepcopy(Mlocal)
    
    for m in range(len(Mlocal)):
        for k in range(len(Mlocal[m])):
            M_updated[m][k] = Mlocal[m][k] - Mout[m][k]
            if k > 0:
                M_updated[m][k] += Mout[m][k - 1]
            elif m > 0:
                M_updated[m][k] += Mout[m - 1][-1]
    
    return M_updated

# Executes DNN Split Partition function for the specified number of iterations. (default=3)
# Each iteration runs for one time slot.
def dnn_per_slot(inputs: dict, iterations=3, save_csv=False):
    # Assumes timeslot is equal to Dmax
    timeslot = inputs["Dmax"]
    
    for cycle in range(iterations):
        print(f"\n===== Time Slot {cycle+1} | Time {round(cycle * timeslot,3)}s =====")

        # 1) alias vehicle counts and zone lengths
        Mm_k_t = inputs["Mm_k_t"]
        zone_lengths = inputs["zone_lengths"]

        # 2) compute per-zone distances for each RSU
        dm_k_t = [
            [ zone_lengths[m] / max(1, Mm_k_t[m][k]) 
              for k in range(len(Mm_k_t[m])) ]
            for m in range(len(Mm_k_t))
        ]

        # 3) compute per-zone speeds for each RSU
        speeds = [
            [ dm_k_t[m][k] / timeslot 
              for k in range(len(dm_k_t[m])) ]
            for m in range(len(dm_k_t))
        ]

        # 4) update vehicle counts via paper’s formula
        Mm_k_t = apply_vm(Mm_k_t, speeds, timeslot, zone_lengths)
        inputs["Mm_k_t"] = Mm_k_t

        # 5) runs dnn partition using the updated distances and vehicle counts.
        results = dnn_partition(
            Mm_k_t,                     
            inputs["Sm_k_bits"],
            dm_k_t,
            inputs["FRm"],
            inputs["vehicle_compute_load"],
            inputs["rsu_compute_load"],
            inputs["Dmax"],
            inputs["Fv"],
            inputs["BR"],
            inputs["Pt_v"],
            inputs["G"],
            inputs["η"],
            inputs["σ2"],
        )

        # 6) Prints results for specified number of iterations
        for idx, df in enumerate(results, start=1):
            print(f"\n--- RSU {idx} ---")
            print(f"Total Vehicles: {df['value'].sum()}")
            print(df.to_string(index=False))
            if save_csv:
                df.to_csv(f"rsu_{idx}_slot_{cycle+1}.csv", index=False)

        # ─── now print diagnostics underneath ───
        print("\n--- Diagnostics ---")
        for m_idx in range(len(dm_k_t)):
            print(
                f"RSU {m_idx+1}: distances = {dm_k_t[m_idx]}, "
                f"post-mobility counts = {Mm_k_t[m_idx]}"
            )

# main function that runs DNN split partition function repeatedly
if __name__ == "__main__":
    inputs = real_world_data()
    dnn_per_slot(inputs, iterations=5, save_csv=False)
