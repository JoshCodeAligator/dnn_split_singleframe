from dnn_split_singleframe import dnn_partition

#INPUTS USED IN DNN PARTITION ALGORITHM
FRm = [1e9, 1.5e9]    # 1 GFLOPS for RSU 
vehicle_compute_load = [0, 18, 33.55, 54.6172, 67.2072, 89.6351, 119.2351]  # Total compute cost for vehicle-side (GFLOPs). size [7] - DNN has 7 layers (N=7)
rsu_compute_load = [119.2351, 101.2351, 85.6851, 64.6179, 52.0279, 29.6, 0] # Total compute cost for RSU-side (GFLOPs). size [7]
Sm_k = [
    [0.5 * 8e6, 0.8 * 8e6, 0.6 * 8e6],  # RSU 1: 3 zones (bits/vehicle)
    [0.7 * 8e6, 0.5 * 8e6]              # RSU 2: 2 zones 
]
d_mk = [
    [30, 40, 50],                       # RSU 1: distances to zones (m)
    [60, 45]                            # RSU 2: distances to zones (m)
]                                       

Mm_k = [
    [3, 5, 4],                          # RSU 1: 3 vehicles in zone 1, 5 in zone 2, 4 in zone 3
    [6, 2]                              # RSU 2: 6 vehicles in zone 1, 2 in zone 2
]

# Runs DNN Algorithm Per RSU for every time slot
def dnn_per_slot(time_slot=0.2, iterations=3):
    for t in range(iterations):
        current_time = round(t * time_slot, 3)
        print(f"\n===== Time Slot {t + 1} | Time: {current_time:.3f}s =====")
        results = dnn_partition(Mm_k, Sm_k, d_mk, FRm, vehicle_compute_load, rsu_compute_load, Dmax=time_slot)

        for idx, df in enumerate(results, start=1):
            print(f"\n--- RSU {idx} ---")
            print(f"Total Vehicles: {df['value'].sum()}")

            #Removes Row Indices for Cleaner Output
            print(df.to_string(index=False)
)

#Displays a Pandas Dataframe 
#Outputs Optimal Values of nm_k_t, a*m_k_t, and b*m_k_t across all zones per RSU for each time slot iteration
dnn_per_slot()


