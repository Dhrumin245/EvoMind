"""
Tests for deterministic behavior
"""
import numpy as np
import random
import torch
from environments.deterministic_env import DeterministicVectorizedArena, DeterministicSeedManager
from core.genome import Genome


def test_deterministic_seeds():
    """Test that seeds produce deterministic results"""
    print("🧪 Testing deterministic seeds...")
    
    # Test 1: Same seed produces same results
    seed = 42
    env1 = DeterministicVectorizedArena(num_envs=10, seed=seed)
    env2 = DeterministicVectorizedArena(num_envs=10, seed=seed)
    
    states1 = env1.reset()
    states2 = env2.reset()
    
    assert np.allclose(states1, states2), "Reset states should be identical"
    print("✓ Reset states are deterministic")
    
    # Test 2: Same actions produce same results
    actions = np.random.randint(0, 3, size=10)
    
    states1, rewards1, dones1 = env1.step(actions)
    states2, rewards2, dones2 = env2.step(actions)
    
    assert np.allclose(states1, states2), "Step states should be identical"
    assert np.allclose(rewards1, rewards2), "Rewards should be identical"
    assert np.all(dones1 == dones2), "Done flags should be identical"
    print("✓ Step results are deterministic")
    
    # Test 3: Different seeds produce different results
    env3 = DeterministicVectorizedArena(num_envs=10, seed=seed + 1)
    states3 = env3.reset()
    
    assert not np.allclose(states1, states3), "Different seeds should produce different results"
    print("✓ Different seeds produce different results")
    
    env1.close()
    env2.close()
    env3.close()
    
    print("✅ All deterministic seed tests passed!")


def test_seed_manager():
    """Test seed manager functionality"""
    print("\n🧪 Testing seed manager...")
    
    manager = DeterministicSeedManager(base_seed=123)
    
    # Test seed generation
    seed1 = manager.get_seed("env")
    seed2 = manager.get_seed("env")
    assert seed1 == seed2, "Same component should get same seed"
    
    seed3 = manager.get_seed("genome")
    assert seed1 != seed3, "Different components should get different seeds"
    
    # Test env seeds
    env_seeds = manager.get_env_seeds(5, "test_envs")
    assert len(env_seeds) == 5, "Should generate correct number of seeds"
    assert len(set(env_seeds)) == 5, "All seeds should be unique"
    
    # Test saving/loading
    manager.save_seeds("test_seeds.json")
    manager2 = DeterministicSeedManager(base_seed=456)
    manager2.load_seeds("test_seeds.json")
    
    assert manager2.get_seed("env") == seed1, "Loaded seeds should match"
    
    import os
    os.remove("test_seeds.json")
    
    print(" Seed manager tests passed!")


def test_deterministic_genome_evaluation():
    """Test deterministic genome evaluation"""
    print("\n Testing deterministic genome evaluation...")

    # Create two identical genomes
    genome1 = Genome(seed=42)
    genome2 = genome1.copy()
    
    # Create deterministic environments with same seed
    env1 = DeterministicVectorizedArena(num_envs=10, seed=42)
    env2 = DeterministicVectorizedArena(num_envs=10, seed=42)
    
    # Run same evaluation
    states1 = env1.reset()
    states2 = env2.reset()
    
    actions1 = genome1.act_batch(states1)
    actions2 = genome2.act_batch(states2)
    
    assert np.all(actions1 == actions2), "Identical genomes should produce same actions"
    
    # Step environments
    states1, rewards1, dones1 = env1.step(actions1)
    states2, rewards2, dones2 = env2.step(actions2)
    
    assert np.allclose(states1, states2), "States should be identical"
    assert np.allclose(rewards1, rewards2), "Rewards should be identical"
    
    env1.close()
    env2.close()
    
    print("✅ Deterministic genome evaluation tests passed!")


def test_async_determinism():
    """Test async environment determinism"""
    print("\n🧪 Testing async environment determinism...")
    
    from core.async_env import AsyncVectorizedArena
    
    # Create async environments with same seed
    async_env1 = AsyncVectorizedArena(num_envs=10, seed=42, num_workers=2)
    async_env2 = AsyncVectorizedArena(num_envs=10, seed=42, num_workers=2)
    
    # Reset
    states1 = async_env1.reset()
    states2 = async_env2.reset()
    
    # Test synchronous step (async internally)
    actions = np.random.randint(0, 3, size=10)
    
    states1, rewards1, dones1 = async_env1.step(actions)
    states2, rewards2, dones2 = async_env2.step(actions)
    
    # Allow small floating point differences due to async timing
    assert np.allclose(states1, states2, rtol=1e-5, atol=1e-5), \
        "Async environments should produce similar states"
    assert np.allclose(rewards1, rewards2, rtol=1e-5, atol=1e-5), \
        "Async environments should produce similar rewards"
    
    async_env1.close()
    async_env2.close()
    
    print("✅ Async environment determinism tests passed!")


def run_all_determinism_tests():
    """Run all determinism tests"""
    print("=" * 60)
    print("Running Determinism Tests")
    print("=" * 60)
    
    try:
        test_deterministic_seeds()
        test_seed_manager()
        test_deterministic_genome_evaluation()
        test_async_determinism()
        
        print("\n" + "=" * 60)
        print("🎉 All determinism tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_determinism_tests()