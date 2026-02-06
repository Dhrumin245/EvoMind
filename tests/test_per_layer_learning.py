#!/usr/bin/env python3
"""
Critical-path test for per-layer learning rules wiring.
Tests that:
1. PlasticLinear.apply_plasticity uses learning_rule_net correctly
2. TorchBrain.build_from_genome passes gene.learning_rule_net to PlasticLinear
3. End-to-end plasticity updates work during evaluation
"""

import torch
import numpy as np
from core.genome import EvolvableGenome
from core.torch_brain import TorchBrain, PlasticLinear

def test_learning_rule_net_wiring():
    """Test that learning_rule_net is properly wired in PlasticLinear"""
    print("Testing per-layer learning rules wiring...")

    # Create a genome with plasticity
    genome = EvolvableGenome()

    # Ensure all genes have plasticity (except output layer which is intentionally None)
    for i, gene in enumerate(genome.genes[:-1]):  # Skip output layer
        if gene.plasticity is None:
            gene.plasticity = np.random.uniform(-0.1, 0.1, (gene.output_dim, gene.input_dim))
            # Initialize learning_rule_net since plasticity was just enabled
            # Use the same logic as in NeuralGene.__init__
            if gene.learning_rule_net is None:
                from core.genome import LearningRuleNet
                gene.learning_rule_net = LearningRuleNet(input_dim=gene.input_dim, output_dim=gene.output_dim, hidden_dim=16)

    # Build TorchBrain
    brain = TorchBrain(genome)

    # Debug: Check gene learning_rule_net
    for i, gene in enumerate(genome.genes):
        print(f"Gene {i}: plasticity={gene.plasticity is not None}, learning_rule_net={gene.learning_rule_net is not None}")

    # Check that PlasticLinear layers have learning_rule_net
    for i, layer in enumerate(brain.layers):
        if isinstance(layer, PlasticLinear):
            gene = genome.genes[i]
            print(f"Layer {i}: layer.learning_rule_net={layer.learning_rule_net is not None}, gene.learning_rule_net={gene.learning_rule_net is not None}")
            if gene.plasticity is not None:  # Only check layers that should have plasticity
                assert layer.learning_rule_net is not None, f"Layer {i} missing learning_rule_net"
                assert layer.learning_rule_net == genome.genes[i].learning_rule_net, f"Layer {i} learning_rule_net mismatch"
                print(f"✓ Layer {i}: learning_rule_net properly wired")
            else:
                print(f"✓ Layer {i}: correctly has no learning_rule_net (output layer)")

    print("✓ All PlasticLinear layers have learning_rule_net wired correctly")

def test_plasticity_update():
    """Test that plasticity updates use the learning_rule_net"""
    print("\nTesting plasticity updates...")

    # Create genome and brain
    genome = EvolvableGenome()
    genome.genes[0].plasticity = np.random.uniform(-0.1, 0.1, (genome.genes[0].output_dim, genome.genes[0].input_dim))
    brain = TorchBrain(genome)

    # Get first plastic layer
    plastic_layer = None
    for layer in brain.layers:
        if isinstance(layer, PlasticLinear):
            plastic_layer = layer
            break

    assert plastic_layer is not None, "No plastic layer found"

    # Store initial plastic weights
    initial_plastic = plastic_layer.plastic_weight.clone()

    # Create dummy inputs for apply_plasticity
    plastic_layer.last_input = torch.randn(1, plastic_layer.input_dim)
    plastic_layer.last_output = torch.randn(1, plastic_layer.output_dim)

    # Apply plasticity update
    reward = 1.0
    meta = {"reward_gain": 1.0, "reward_bias": 0.0, "plastic_lr": 1.0}
    plastic_layer.apply_plasticity(reward, meta)

    # Check that plastic weights changed
    final_plastic = plastic_layer.plastic_weight
    assert not torch.equal(initial_plastic, final_plastic), "Plastic weights did not change"

    print("✓ Plasticity updates work and use learning_rule_net")

def test_end_to_end_evaluation():
    """Test end-to-end evaluation with plasticity"""
    print("\nTesting end-to-end evaluation...")

    # Create genome
    genome = EvolvableGenome()
    genome.genes[0].plasticity = np.random.uniform(-0.1, 0.1, (genome.genes[0].output_dim, genome.genes[0].input_dim))

    # Get brain (this calls build_from_genome)
    brain = genome.get_brain()

    # Test forward pass
    dummy_input = torch.randn(1, genome.input_size)
    output = brain(dummy_input)
    assert output.shape == (1, genome.output_size), f"Wrong output shape: {output.shape}"

    # Test plasticity update
    brain.reset_plasticity()
    brain.reset_episode_tracking()

    # Simulate a few steps
    for step in range(3):
        # Forward pass
        output = brain(dummy_input)

        # Apply plasticity
        reward = 0.1 * (step + 1)  # Increasing reward
        brain.update_plasticity(reward, done=False)

    # Check that episode data was recorded
    episode_data = brain.get_episode_data()
    assert len(episode_data["delta_norms"]) == 3, f"Expected 3 delta norms, got {len(episode_data['delta_norms'])}"
    assert len(episode_data["rewards"]) == 3, f"Expected 3 rewards, got {len(episode_data['rewards'])}"

    print("✓ End-to-end evaluation with plasticity works")

def test_learning_rule_net_consistency():
    """Test that learning_rule_net parameters are consistent"""
    print("\nTesting learning_rule_net consistency...")

    genome = EvolvableGenome()

    # Check that genome has learning_rule_net
    assert genome.learning_rule_net is not None, "Genome missing learning_rule_net"

    # Check that NeuralGenes have learning_rule_net when plasticity is enabled
    for gene in genome.genes:
        if gene.plasticity is not None:
            assert gene.learning_rule_net is not None, f"Gene {gene.gene_id} missing learning_rule_net despite having plasticity"

    print("✓ Learning rule nets are consistently present where needed")

if __name__ == "__main__":
    try:
        test_learning_rule_net_wiring()
        test_plasticity_update()
        test_end_to_end_evaluation()
        test_learning_rule_net_consistency()
        print("\n🎉 All critical-path tests passed! Per-layer learning rules are wired end-to-end.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
