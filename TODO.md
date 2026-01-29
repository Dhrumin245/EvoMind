# TODO: Replace Dict-Based Learning Rule with Learned Function

## Steps to Complete

- [x] Modify LearningRuleNet to accept pre_dim, post_dim, compute input/output dims, and change forward to take pre, post, reward, w, t and return ΔW.
- [x] Add learning_rule_net to NeuralGene, with initialization, copying, serialization, and mutation.
- [x] In EvolvableGenome, create learning_rule_net for plastic genes.
- [x] Modify PlasticLinear to include learning_rule_net and timestep, update apply_plasticity to use the net for ΔW.
- [x] Update TorchBrain.build_from_genome to pass gene.learning_rule_net into PlasticLinear.
- [x] Wire per-layer learning rules end-to-end.
- [x] Update crossover and mutation for the new nets (if needed).
- [x] Test the implementation to ensure plasticity updates work correctly.
- [x] Verify that the nets are evolved and mutated properly.
