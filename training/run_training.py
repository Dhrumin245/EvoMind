"""
Entry point for running the evolution training
"""

import sys
import argparse
from main import main as train_main

def parse_args():
    parser = argparse.ArgumentParser(description='Evolution Arena Training')
    
    parser.add_argument('--population', type=int, default=100,
                       help='Population size')
    parser.add_argument('--generations', type=int, default=1000,
                       help='Number of generations')
    parser.add_argument('--envs', type=int, default=64,
                       help='Environments per genome')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint file to load')
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Starting Evolution Arena Training")
    print(f"Population: {args.population}")
    print(f"Generations: {args.generations}")
    print(f"Environments per genome: {args.envs}")

    train_main()
