# NeuroGenesis PPO Conflict Fix - Implementation Plan

## Current Status

- PPO is already disabled in main.py (enable_ppo_inner_loop = False)
- Dead neuron warnings are silenced in torch_brain.py
- Basic fitness penalties exist for dead units

## Required Changes

### 1. Add Neural Health Controller to TorchBrain

- [ ] Create NeuralHealthController class in torch_brain.py
- [ ] Detect dead neurons per layer (activation ratio < 1e-6)
- [ ] Detect saturated neurons per layer
- [ ] Track neural health metrics across episodes
- [ ] Trigger architecture mutations when health is poor

### 2. Enhance Fitness Computation

- [ ] Modify compute_fitness_from_metrics in main.py
- [ ] Add heavy penalties for dead neurons (multiplicative penalty)
- [ ] Add penalties for saturated neurons
- [ ] Make dead neurons strongly reduce fitness to create evolutionary pressure

### 3. Stabilize Plasticity Updates

- [ ] Review PlasticLinear.apply_plasticity in torch_brain.py
- [ ] Ensure plasticity doesn't conflict with base weights
- [ ] Add stability monitoring and adaptive clamping
- [ ] Prevent runaway plasticity that causes dead neurons

### 4. Add Architecture Mutation Triggers

- [ ] When dead neurons detected, trigger mutations:
  - [ ] Skip connection insertion
  - [ ] Activation function mutation
  - [ ] Layer reinitialization
- [ ] Make mutations more aggressive for unhealthy networks

### 5. Update Logging and Monitoring

- [ ] Replace silenced warnings with evolutionary health summaries
- [ ] Log neural health metrics per generation
- [ ] Track recovery from dead neuron states

## Testing

- [ ] Verify dead neurons are penalized in fitness
- [ ] Check that unhealthy networks mutate more aggressively
- [ ] Ensure plasticity updates don't cause dead neurons
- [ ] Confirm no PPO conflicts remain
