import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import warnings

# Use DeterministicVectorizedArena for true vectorization
from environments.deterministic_env import DeterministicVectorizedArena
from curriculum.curriculum import CurriculumStage


class BatchedGenomeEvaluator:
    """
    Handles batched evaluation of genomes with proper GPU batching.
    Supports both CPU and GPU evaluation with efficient kernel launches.
    """
    
    def __init__(self,
                 num_envs_per_genome: int = 4,
                 use_gpu: bool = False,
                 max_workers: Optional[int] = 2,
                 curriculum_stage: CurriculumStage = CurriculumStage.FORAGING,
                 curriculum_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the batched evaluator.
        
        Args:
            num_envs_per_genome: Number of environments per genome evaluation
            use_gpu: Whether to use GPU for evaluation
            max_workers: Maximum number of worker processes (None = CPU count)
            curriculum_stage: Current curriculum stage
            curriculum_config: Stage-specific configuration
        """
        self.num_envs_per_genome = num_envs_per_genome
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.max_workers = max_workers or mp.cpu_count()
        self.curriculum_stage = curriculum_stage
        self.curriculum_config = curriculum_config or {}
        
        # Device setup
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Statistics
        self.eval_count = 0
        self.total_steps = 0
        
    def evaluate_genomes_batch_gpu(self, genomes: List, 
                                   seed_offset: int = 0) -> np.ndarray:
        """
        Evaluate multiple genomes with true GPU batching.
        All genomes are evaluated in a single forward pass.
        
        Args:
            genomes: List of genome objects with neural networks
            seed_offset: Base seed for deterministic evaluation
            
        Returns:
            Array of fitness scores for each genome
        """
        if not genomes:
            return np.array([])
        
        num_genomes = len(genomes)
        total_envs = num_genomes * self.num_envs_per_genome
        
        # Create a single arena for all environments
        arena = DeterministicVectorizedArena(
            num_envs=total_envs,
            max_steps=1000,
            seed=seed_offset,
            stage_config=self.curriculum_config
        )
        
        # Reset all environments
        states = arena.reset()
        states_tensor = torch.from_numpy(states).float().to(self.device)
        
        # Initialize tracking arrays
        episode_rewards = torch.zeros(num_genomes, device=self.device, dtype=torch.float32)
        episode_counts = torch.zeros(num_genomes, device=self.device, dtype=torch.float32)
        active_mask = torch.ones(total_envs, device=self.device, dtype=torch.bool)
        
        # Convert genomes to batched neural network
        # This assumes genomes have a method to extract weights for batched processing
        batched_weights = self._prepare_batched_weights(genomes)
        
        # Create batched neural network
        batched_nn = self._create_batched_nn(batched_weights, genomes[0])
        
        step = 0
        max_steps = arena.max_steps
        
        while step < max_steps and torch.any(active_mask):
            # Forward pass through batched neural network
            with torch.no_grad():
                actions = batched_nn(states_tensor)
            
            # Convert actions to numpy for environment step
            actions_np = actions.cpu().numpy()
            
            # Step all environments
            states_np, rewards_np, dones_np = arena.step(actions_np)
            
            # Convert to tensors
            states_tensor = torch.from_numpy(states_np).float().to(self.device)
            rewards_tensor = torch.from_numpy(rewards_np).float().to(self.device)
            dones_tensor = torch.from_numpy(dones_np).bool().to(self.device)
            
            # Update active mask
            active_mask = active_mask & (~dones_tensor)
            
            # Accumulate rewards ONLY for active environments
            active_rewards = rewards_tensor * active_mask.float()
            
            # Sum rewards per genome
            for i in range(num_genomes):
                start_idx = i * self.num_envs_per_genome
                end_idx = (i + 1) * self.num_envs_per_genome
                
                # Get rewards for this genome's environments
                genome_active = active_mask[start_idx:end_idx]
                genome_rewards = active_rewards[start_idx:end_idx]
                
                # Only accumulate if any environment is active
                if torch.any(genome_active):
                    # Average reward per active environment
                    episode_rewards[i] += torch.sum(genome_rewards) / torch.sum(genome_active.float())
                    episode_counts[i] += 1
            
            step += 1
            self.total_steps += total_envs
        
        # Calculate final fitness as mean reward per environment
        fitness_scores = np.zeros(num_genomes, dtype=np.float32)
        for i in range(num_genomes):
            if episode_counts[i] > 0:
                # Average across all steps and environments
                fitness_scores[i] = (episode_rewards[i] / episode_counts[i]).item()
            else:
                fitness_scores[i] = 0.0
        
        # Update genome fitness
        for i, genome in enumerate(genomes):
            genome.fitness = float(fitness_scores[i])
        
        self.eval_count += num_genomes
        
        return fitness_scores
    
    def evaluate_genomes_cpu_parallel(self, genomes: List, 
                                      seed_offset: int = 0) -> np.ndarray:
        """
        Evaluate genomes using CPU multiprocessing.
        Each genome is evaluated in a separate process.
        
        Args:
            genomes: List of genome objects
            seed_offset: Base seed for deterministic evaluation
            
        Returns:
            Array of fitness scores
        """
        if not genomes:
            return np.array([])
        
        # Prepare arguments for workers
        args = []
        for i, genome in enumerate(genomes):
            seed = seed_offset + i * 1000 if seed_offset is not None else None
            args.append((
                genome,
                self.num_envs_per_genome,
                self.curriculum_stage,
                self.curriculum_config,
                seed
            ))
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._evaluate_single_genome_worker, args))
        
        # Update genome fitness
        fitness_scores = []
        for i, genome in enumerate(genomes):
            result = results[i]
            if isinstance(result, dict):
                fitness = result.get("fitness", -1)
                error = result.get("error")
                if error:
                    print(f"Genome {i} evaluation error: {error}")
            else:
                fitness = result
            genome.fitness = float(fitness)
            fitness_scores.append(fitness)

        self.eval_count += len(genomes)

        return np.array(fitness_scores)
    
    @staticmethod
    def _evaluate_single_genome_worker(args):
        """
        Worker function for single genome evaluation.
        Must be static for multiprocessing.
        """
        genome, num_envs, stage, config, seed = args
        
        try:
            # Use brain from genome (cached)
            brain = genome.get_brain()
            
            # Create deterministic arena
            arena = DeterministicVectorizedArena(
                num_envs=num_envs,
                max_steps=1000,
                seed=seed,
                stage_config=config
            )
            
            # Reset environment
            states = arena.reset()
            
            # Initialize tracking
            total_reward = np.zeros(num_envs, dtype=np.float32)
            active = np.ones(num_envs, dtype=bool)
            
            step = 0
            max_steps = arena.max_steps
            
            # Run episode
            while step < max_steps and np.any(active):
                # Get actions from genome
                actions = genome.act_batch(states)
                
                # Step environment
                states, rewards, dones = arena.step(actions)
                
                # Update active mask
                active = active & (~dones)
                
                # Accumulate rewards only for active environments
                total_reward[active] += rewards[active]
                
                step += 1
            
            # Calculate fitness as mean reward per environment
            if np.any(active):  # Some environments may have finished
                # Average reward across all environments
                fitness = np.mean(total_reward)
            else:
                fitness = np.mean(total_reward)
            
            return {"fitness": float(fitness), "error": None}
            
        except Exception as e:
            print(f"Error in worker (PID {os.getpid()}): {e}")
            return {"fitness": -1, "error": str(e)}
    
    def _prepare_batched_weights(self, genomes: List) -> torch.Tensor:
        """
        Prepare batched weights from multiple genomes for efficient GPU processing.
        
        Args:
            genomes: List of genome objects
            
        Returns:
            Batched weights tensor of shape [num_genomes, total_weights]
        """
        # Extract weights from each genome
        weight_list = []
        for genome in genomes:
            # Assuming genome has a get_weights() method that returns a numpy array
            if hasattr(genome, 'get_weights'):
                weights = genome.get_weights()
            elif hasattr(genome, 'weights'):
                weights = genome.weights
            else:
                raise AttributeError(f"Genome {type(genome)} has no weights attribute")
            
            weight_list.append(weights)
        
        # Convert to batched tensor
        batched_weights = torch.stack([torch.from_numpy(w).float() for w in weight_list])
        
        if self.use_gpu:
            batched_weights = batched_weights.cuda()
        
        return batched_weights
    
    def _create_batched_nn(self, batched_weights: torch.Tensor, 
                          template_genome) -> nn.Module:
        """
        Create a batched neural network from template genome architecture.
        
        Args:
            batched_weights: Batched weights tensor [num_genomes, total_weights]
            template_genome: Genome instance for architecture template
            
        Returns:
            Batched neural network module
        """
        # Get neural network architecture from template genome
        if hasattr(template_genome, 'get_nn_architecture'):
            input_size, hidden_sizes, output_size = template_genome.get_nn_architecture()
        else:
            # Default architecture (should match genome.py)
            input_size = 7  # From VectorizedArena state dimension
            hidden_sizes = [16, 16]  # Default hidden layers
            output_size = 3  # Continuous action vector [turn, thrust, special]
        
        # Create batched neural network
        batched_nn = BatchedNeuralNetwork(
            batch_size=batched_weights.shape[0],
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size
        )
        
        # Load weights into the batched network
        batched_nn.load_batched_weights(batched_weights)
        
        if self.use_gpu:
            batched_nn = batched_nn.cuda()
        
        return batched_nn
    
    def update_curriculum(self, stage: CurriculumStage, 
                         config: Optional[Dict[str, Any]] = None):
        """
        Update curriculum configuration for evaluation.
        
        Args:
            stage: New curriculum stage
            config: Optional configuration overrides
        """
        self.curriculum_stage = stage
        if config:
            self.curriculum_config.update(config)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        return {
            'evaluations': self.eval_count,
            'total_steps': self.total_steps,
            'envs_per_genome': self.num_envs_per_genome,
            'using_gpu': self.use_gpu,
            'curriculum_stage': self.curriculum_stage.name
        }


class BatchedNeuralNetwork(nn.Module):
    """
    Batched neural network that processes multiple sets of weights simultaneously.
    Supports efficient GPU batching for thousands of genomes.
    """
    
    def __init__(self, batch_size: int, input_size: int, 
                 hidden_sizes: List[int], output_size: int):
        """
        Initialize batched neural network.
        
        Args:
            batch_size: Number of parallel weight sets (genomes)
            input_size: Input dimension
            hidden_sizes: List of hidden layer sizes
            output_size: Output dimension
        """
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        
        # Build layer dimensions
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1
        
        # Create batched parameters
        self.batched_weights = nn.ParameterList()
        self.batched_biases = nn.ParameterList()
        
        for i in range(self.num_layers):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            
            # Batched weights: [batch_size, out_features, in_features]
            weights = nn.Parameter(
                torch.randn(batch_size, out_features, in_features) * 0.1
            )
            
            # Batched biases: [batch_size, out_features]
            biases = nn.Parameter(
                torch.zeros(batch_size, out_features)
            )
            
            self.batched_weights.append(weights)
            self.batched_biases.append(biases)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through batched network.
        
        Args:
            x: Input tensor of shape [total_envs, input_size]
            
        Returns:
            Output tensor of shape [total_envs, output_size]
        """
        # Reshape input: [batch_size, envs_per_genome, input_size]
        batch_size = self.batch_size
        envs_per_genome = x.shape[0] // batch_size
        
        x_reshaped = x.view(batch_size, envs_per_genome, self.input_size)
        
        # Apply batched layers
        for i in range(self.num_layers):
            # Batched matrix multiplication
            # x_reshaped: [batch_size, envs_per_genome, in_features]
            # weights: [batch_size, out_features, in_features]
            # Result: [batch_size, envs_per_genome, out_features]
            x_reshaped = torch.bmm(
                x_reshaped,
                self.batched_weights[i].transpose(1, 2)  # [batch_size, in_features, out_features]
            )
            
            # Add bias
            x_reshaped = x_reshaped + self.batched_biases[i].unsqueeze(1)
            
            # Apply activation (except last layer)
            if i < self.num_layers - 1:
                x_reshaped = torch.tanh(x_reshaped)
            else:
                # Output layer: tanh for actions in [-1, 1]
                x_reshaped = torch.tanh(x_reshaped)
        
        # Reshape back: [total_envs, output_size]
        return x_reshaped.view(-1, self.output_size)
    
    def load_batched_weights(self, weights_tensor: torch.Tensor):
        """
        Load batched weights from a flat tensor.
        
        Args:
            weights_tensor: Flat tensor of shape [batch_size, total_weights]
        """
        # This method needs to know how to split the flat weights into layer weights
        # Implementation depends on genome encoding scheme
        # For now, we'll use a simple heuristic
        
        batch_size = weights_tensor.shape[0]
        if batch_size != self.batch_size:
            raise ValueError(f"Expected batch_size {self.batch_size}, got {batch_size}")
        
        # Calculate total expected weights
        total_params = 0
        for i in range(self.num_layers):
            in_features = self.batched_weights[i].shape[2]
            out_features = self.batched_weights[i].shape[1]
            total_params += in_features * out_features + out_features
        
        if weights_tensor.shape[1] != total_params:
            raise ValueError(f"Expected {total_params} weights, got {weights_tensor.shape[1]}")
        
        # Simple loading: split equally (this should match genome encoding)
        # In practice, this should be coordinated with Genome class
        idx = 0
        for i in range(self.num_layers):
            in_features = self.batched_weights[i].shape[2]
            out_features = self.batched_weights[i].shape[1]
            
            # Extract weights
            weight_params = in_features * out_features
            flat_weights = weights_tensor[:, idx:idx+weight_params]
            idx += weight_params
            
            # Reshape to [batch_size, out_features, in_features]
            reshaped_weights = flat_weights.view(batch_size, out_features, in_features)
            self.batched_weights[i].data.copy_(reshaped_weights)
            
            # Extract biases
            bias_params = out_features
            flat_biases = weights_tensor[:, idx:idx+bias_params]
            idx += bias_params
            
            self.batched_biases[i].data.copy_(flat_biases)


def evaluate_genomes_parallel(genomes: List,
                              num_envs_per_genome: int = 64,
                              use_gpu: bool = False,
                              max_workers: Optional[int] = None,
                              curriculum_stage: CurriculumStage = CurriculumStage.FORAGING,
                              curriculum_config: Optional[Dict[str, Any]] = None,
                              seed_offset: int = 0) -> np.ndarray:
    """
    Main entry point for parallel genome evaluation.
    
    Args:
        genomes: List of genome objects to evaluate
        num_envs_per_genome: Number of parallel environments per genome
        use_gpu: Whether to use GPU acceleration
        max_workers: Maximum CPU workers (for CPU mode)
        curriculum_stage: Current curriculum stage
        curriculum_config: Stage configuration
        seed_offset: Base seed for deterministic evaluation
        
    Returns:
        Array of fitness scores
    """
    evaluator = BatchedGenomeEvaluator(
        num_envs_per_genome=num_envs_per_genome,
        use_gpu=use_gpu,
        max_workers=max_workers,
        curriculum_stage=curriculum_stage,
        curriculum_config=curriculum_config
    )
    
    if use_gpu and torch.cuda.is_available():
        # Use batched GPU evaluation
        fitness_scores = evaluator.evaluate_genomes_batch_gpu(
            genomes, seed_offset=seed_offset
        )
    else:
        # Use CPU multiprocessing
        fitness_scores = evaluator.evaluate_genomes_cpu_parallel(
            genomes, seed_offset=seed_offset
        )
    
    return fitness_scores