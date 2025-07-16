import math
import pandas as pd
import numpy as np

# Output Data from numpy file
outputs = np.load("outputVehNum_all.npy")

# Data for inputs are obtained from real world trace.
def collected_data():
    vcl = [0, 18e9, 33.55e9, 54.6172e9, 67.2072e9, 89.6351e9, 119.2351e9]       # vehicle compute load for each DNN partition point. (7 partition points)
    rcl = [119.2351e9, 101.2351e9, 85.6851e9, 64.6179e9, 52.0279e9, 29.6e9, 0]  # rsu compute load for each DNN partition point. (7 partition points)
    Sm_k = [32.8e6, 16.4e6, 8.2e6, 4.1e6, 2.05e6, 2.05e6, 1.02e6]               # (bytes). Real feature map data size per DNN layer (1-7) from YOLOv5x model.
    
    timeslot = 0.05                                                             # 50 ms
    total_slots = len(outputs)

    RSU = 3
    zone_split = 5
    
    return {
        "Fv": 10e9,      # Compute Power for vehicles - 10 GFLOPS.
        "FRm": 100e9,    # Compute Power for RSUs (100 GFLOPS)
        "BR": 100e6,     # 100 MHz LTE uplink
        "Dmax": 0.2,     # 20 ms max delay. 
        "timeslot": timeslot,
        "total_slots": total_slots,
        "Pt_v": 0.1,     # 0.1 Watts 
        "G": 1.0,        
        "η": 3.0,        # Path loss exponent
        "σ2": 1e-13,     # 1e-13 Watts
        "rsu": RSU,
        "zones": zone_split,
        
        "vehicle_compute_load": vcl,
        "rsu_compute_load": rcl,

        # Converting feature size Sm_k from bytes to bits.
        "Sm_k_bits": [x * 8 for x in Sm_k],
    }

inputs = collected_data()

def genetic(time_slot=0):
    selected_row = outputs[time_slot]
    vehicle_zones = selected_row.reshape(inputs["rsu"], inputs["zones"])


