# Brax PPO with Trac optimizer support and dormant neuron tracking
"""
Local copy of Brax PPO training adapted for:
1. Trac optimizer compatibility (passes params to optimizer.update)
2. Dormant neuron tracking and ReDo (Reinitializing Dormant neurons)

Based on inspiration/brax_wrapper with fixed imports.
"""

from brax_trac.ppo_train import train
