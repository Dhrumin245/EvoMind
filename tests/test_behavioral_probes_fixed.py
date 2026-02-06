#!/usr/bin/env python3
"""
Test script for behavioral probes functionality.
Demonstrates the behavioral probe suite on evolved genomes.
"""

import numpy as np
import time
from core.genome import EvolvableGenome
from evaluation.behavioral_probes import BehavioralProbe

def test_behavioral_probes():
    """Test the behavioral probe suite on sample genomes"""
    print("Testing Behavioral Probe Suite")
    print("=" * 50)

    # Create a few test genomes
    print("Creating test genomes...")
    genomes = []
    for i in range(3):
        genome = EvolvableGenome(
            genome_id=f"test_genome_{i}",
            input_size=6,
            output_size=4,
            init_modules=2,
            init_neurons=8,
            seed=i
        )
        # Initialize the brain for each genome
        genome.get_brain()
        genomes.append(genome)

    print(f"Created {len(genomes)} test genomes")

    # Run behavioral probe suite on each genome
    print("\nRunning behavioral probe suite...")
    start_time = time.time()

    for i, genome in enumerate(genomes):
        print(f"\n--- Testing Genome {genome.genome_id} ---")

        # Run full diagnostic suite
        report = BehavioralProbe.run_diagnostic_suite(genome, generation=0)

        print(f"Overall Score: {report.summary_scores['overall_score']:.3f}")
        print(f"Behavioral Profile: {report.behavioral_profile['archetype']}")
        print("Key Capabilities:")
        for cap, score in report.behavioral_profile['cognitive_capabilities'].items():
            print(f"  {cap}: {score:.3f}")

    total_time = time.time() - start_time
    print(".2f")

    # Test individual probe methods
    print("\nTesting individual probe methods...")
    test_genome = genomes[0]

    # Memory capacity test
    memory_result = BehavioralProbe.test_memory_capacity(test_genome, sequence_length=5)
    print(f"Memory Capacity Score: {memory_result.score:.3f}")

    # Generalization test
    generalization_result = BehavioralProbe.test_generalization(
        test_genome,
        train_envs=['normal', 'noisy'],
        test_envs=['sparse', 'dense']
    )
    print(f"Generalization Score: {generalization_result.score:.3f}")

    # Learning speed test
    learning_result = BehavioralProbe.test_learning_speed(test_genome, 'reversed_rewards')
    print(f"Learning Speed Score: {learning_result.score:.3f}")

    # Credit assignment test
    credit_result = BehavioralProbe.test_credit_assignment(test_genome, 'medium_delay')
    print(f"Credit Assignment Score: {credit_result.score:.3f}")

    # Generate and save probe report
    print("\nGenerating comprehensive probe report...")
    report_json = BehavioralProbe.generate_probe_report(test_genome, generation=0)
    print("Report generated successfully!")

    # Test population-level integration
    print("\nTesting population-level integration...")
    population_results = BehavioralProbe.integrate_with_evaluation_pipeline(
        genomes, generation=0, save_reports=False
    )

    print(f"Population Summary: {len(population_results['individual_reports'])} genomes probed")
    print(f"Average Overall Score: {population_results['population_summary']['population_overall_mean']:.3f}")

    print("\n✅ Behavioral probe testing completed successfully!")

if __name__ == "__main__":
    test_behavioral_probes()
