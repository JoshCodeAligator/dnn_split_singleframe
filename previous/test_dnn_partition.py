'''
import pytest
from dnn_split_optimizer import dnn_per_slot, real_world_data, apply_vm
from dnn_split_singleframe import dnn_partition
import copy
import pandas as pd

# Unit Tests
# Verify that individual functions behave as expected in isolation.
def test_default_execution_structure():
    """Test that default inputs produce valid DataFrames with expected columns."""
    inputs = real_world_data()
    results = dnn_partition(
        inputs["Mm_k"], inputs["Sm_k"], inputs["d_mk"], inputs["FRm"],
        inputs["vehicle_compute_load"], inputs["rsu_compute_load"], inputs["Dmax"],
        inputs["Fv"], inputs["BR"], inputs["Pt_v"], inputs["G"], inputs["η"], inputs["σ2"]
    )
    assert all(not df.empty for df in results), "Some output DataFrames are unexpectedly empty."
    for df in results:
        for col in ["zone", "n", "value", "alpha*", "beta*", "total_delay"]:
            assert col in df.columns, f"Missing column {col}"

def test_zero_vehicle_input_behavior():
    """Ensure that zero vehicles lead to no delay and split index 0."""
    inputs = real_world_data()
    inputs["Mm_k"] = [[0 for _ in rsu] for rsu in inputs["Mm_k"]]
    results = dnn_partition(
        inputs["Mm_k"], inputs["Sm_k"], inputs["d_mk"], inputs["FRm"],
        inputs["vehicle_compute_load"], inputs["rsu_compute_load"], inputs["Dmax"],
        inputs["Fv"], inputs["BR"], inputs["Pt_v"], inputs["G"], inputs["η"], inputs["σ2"]
    )
    for df in results:
        assert (df["value"] == 0).all()
        assert (df["n"] == 0).all()

def test_vehicle_mobility_distribution():
    """Test that vehicle mobility redistributes vehicles but conserves total."""
    inputs = real_world_data()
    original = copy.deepcopy(inputs["Mm_k"])
    moved = apply_vm(original, mobility_ratio=0.4)
    assert moved != original, "Mobility application didn't affect distribution."
    assert sum(sum(x) for x in moved) == sum(sum(x) for x in original)

def test_apply_vm_default_behavior():
    """Default mobility ratio test (0.2)."""
    original = [[5, 5], [4, 6]]
    moved = apply_vm(original)
    assert original != moved
    assert sum(map(sum, moved)) == sum(map(sum, original))

def test_vehicle_counts_never_negative():
    """Ensure vehicle mobility does not cause negative zone counts."""
    inputs = real_world_data()
    inputs["Mm_k"] = [[0, 1], [1, 0]]
    updated = apply_vm(inputs["Mm_k"], mobility_ratio=0.95)
    for rsu in updated:
        for zone in rsu:
            assert zone >= 0, "Negative vehicle count after mobility"

def test_high_compute_prefers_vehicle_side():
    """High compute & bandwidth should allow at least one zone to run full or near-full on vehicle side."""
    inputs = real_world_data()
    inputs["Fv"] = 1e13         # 10 TFLOPS for Vehicle
    inputs["BR"] = 50e9         # Increased bandwidth to 50 Gbps 
    inputs["Dmax"] = 2.0        # High Delay Constraint

    inputs["Mm_k"] = [
        [5, 5, 5],  # Increases vehicle count to 5 in RSU 1
        [5, 5]      # Increases vehicle count to 5 in RSU 2
    ]
    inputs["FRm"] = [100e9, 100e9]  # Decreases RSU Compute Power to 100 GFLOPs per RSU
    
    results = dnn_partition(
        inputs["Mm_k"], inputs["Sm_k"], inputs["d_mk"], inputs["FRm"],
        inputs["vehicle_compute_load"], inputs["rsu_compute_load"], inputs["Dmax"],
        inputs["Fv"], inputs["BR"], inputs["Pt_v"], inputs["G"], inputs["η"], inputs["σ2"]
    )
    #Checks if there are any high dnn splits (n >= 5) in results. Test case fails if there aren't any present in at least one zone. 
    high_splits = any((df["n"] >= 5).any() for df in results)
    assert high_splits, "Expected at least one zone with split n ≥ 5 under high resources"

def test_low_delay_forces_split_simplicity():
    """Tight delay budget should force simpler/lower DNN splits."""
    inputs = real_world_data()
    inputs["Dmax"] = 0.1
    results = dnn_partition(
        inputs["Mm_k"], inputs["Sm_k"], inputs["d_mk"], inputs["FRm"],
        inputs["vehicle_compute_load"], inputs["rsu_compute_load"], inputs["Dmax"],
        inputs["Fv"], inputs["BR"], inputs["Pt_v"], inputs["G"], inputs["η"], inputs["σ2"]
    )
    for df in results:
        assert df["total_delay"].max() <= 0.1, "Some zones violate delay constraint"

# Integration Tests
# Test interaction between multiple components/modules to verify they work together
def test_optimizer_runs_multiple_iterations():
    """Run full optimizer for multiple time slots (integration test)."""
    inputs = real_world_data()
    inputs["Mm_k"] = [[4, 2], [1, 3]]
    dnn_per_slot(inputs, iterations=2)

def test_optimizer_with_variable_mobility():
    """Test system behavior under different mobility ratios."""
    base = real_world_data()
    for ratio in [0.0, 0.2, 0.7]:
        inputs = copy.deepcopy(base)
        dnn_per_slot(inputs, iterations=1, mobility_ratio=ratio)

def test_full_mobility_impact():
    """Test extreme case where all vehicles are mobile."""
    inputs = real_world_data()
    moved = apply_vm(inputs["Mm_k"], mobility_ratio=1.0)
    assert all(any(z > 0 for z in rsu) for rsu in moved), "Mobility should redistribute vehicles"

# Edge Case and Stress Tests
# Test unusual or extreme scenarios to ensure robustness
def test_all_rsu_same_processing_capacity():
    """Identical RSU compute capacities should still produce valid results."""
    inputs = real_world_data()
    inputs["FRm"] = [300e9 for _ in inputs["FRm"]]
    results = dnn_partition(
        inputs["Mm_k"], inputs["Sm_k"], inputs["d_mk"], inputs["FRm"],
        inputs["vehicle_compute_load"], inputs["rsu_compute_load"], inputs["Dmax"],
        inputs["Fv"], inputs["BR"], inputs["Pt_v"], inputs["G"], inputs["η"], inputs["σ2"]
    )
    for df in results:
        assert isinstance(df, pd.DataFrame)

def test_extreme_rsu_imbalance():
    """Extreme RSU imbalance should not crash the optimizer and yield at least one valid output."""
    inputs = real_world_data()
    inputs["FRm"] = [800e9, 1e9]  # One powerful, one weak RSU
    results = dnn_partition(
        inputs["Mm_k"], inputs["Sm_k"], inputs["d_mk"], inputs["FRm"],
        inputs["vehicle_compute_load"], inputs["rsu_compute_load"], inputs["Dmax"],
        inputs["Fv"], inputs["BR"], inputs["Pt_v"], inputs["G"], inputs["η"], inputs["σ2"]
    )
    assert len(results) >= 1, "At least one RSU should produce a result"
    assert all(isinstance(df, pd.DataFrame) for df in results)


'''