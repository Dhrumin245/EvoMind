#!/usr/bin/env python3
"""Test script for Milestone 1: Lineage + mutation history (Genome metadata)"""

from genome import EvolvableGenome
import json

def test_genome_metadata():
    """Test genome metadata functionality"""
    print("Testing Genome Metadata for Milestone 1")
    print("=" * 50)

    # Create a test genome
    genome = EvolvableGenome(genome_id='test_genome', input_size=6, output_size=4)

    # Test metadata initialization
    print("1. Initial metadata:")
    print(f"   Parent IDs: {genome.metadata.parent_ids}")
    print(f"   Birth generation: {genome.metadata.birth_generation}")
    print(f"   Mutation history: {genome.metadata.mutation_history}")

    # Test set_parents
    genome.set_parents(['parent1', 'parent2'], 5)
    print("\n2. After set_parents:")
    print(f"   Parent IDs: {genome.metadata.parent_ids}")
    print(f"   Birth generation: {genome.metadata.birth_generation}")

    # Test record_mutation
    genome.record_mutation({'type': 'weight_mutation', 'layer': 0, 'strength': 0.1})
    genome.record_mutation({'type': 'architecture_mutation', 'operation': 'add_neuron'})
    print("\n3. After record_mutation:")
    print(f"   Mutation history length: {len(genome.metadata.mutation_history)}")
    for i, mutation in enumerate(genome.metadata.mutation_history):
        print(f"   Mutation {i}: {mutation}")

    # Test serialization
    data = genome.to_dict()
    has_metadata = 'metadata' in data
    print(f"\n4. Serialization includes metadata: {has_metadata}")

    if has_metadata:
        metadata = data['metadata']
        print(f"   Serialized parent_ids: {metadata.get('parent_ids')}")
        print(f"   Serialized birth_generation: {metadata.get('birth_generation')}")
        print(f"   Serialized mutation_history length: {len(metadata.get('mutation_history', []))}")

    # Test deserialization
    genome2 = EvolvableGenome.from_dict(data)
    print("\n5. Deserialized metadata matches:")
    print(f"   Parent IDs: {genome2.metadata.parent_ids}")
    print(f"   Birth generation: {genome2.metadata.birth_generation}")
    print(f"   Mutation history length: {len(genome2.metadata.mutation_history)}")

    # Verify data integrity
    parents_match = genome.metadata.parent_ids == genome2.metadata.parent_ids
    birth_gen_match = genome.metadata.birth_generation == genome2.metadata.birth_generation
    history_len_match = len(genome.metadata.mutation_history) == len(genome2.metadata.mutation_history)

    print("
6. Data integrity check:")
    print(f"   Parents match: {parents_match}")
    print(f"   Birth generation match: {birth_gen_match}")
    print(f"   History length match: {history_len_match}")

    all_good = parents_match and birth_gen_match and history_len_match and has_metadata
    print(f"\n✅ Milestone 1 implementation: {'PASS' if all_good else 'FAIL'}")

    return all_good

if __name__ == "__main__":
    test_genome_metadata()
