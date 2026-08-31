# Long-Horizon LLM Agent Post-Training for ARC-AGI-3

This project studies how post-training can improve LLM agents in long-horizon, multi-turn environments with sparse feedback. ARC-AGI-3 serves as the primary testbed: an agent must infer visual rules from interaction, maintain state across a growing trajectory, and select actions whose consequences may only become clear many steps later.

The project began with a zero-shot **Recursive Language Model (RLM)** baseline and later evolved into an end-to-end reinforcement-learning pipeline built on **SkyRL**.

> **Status:** The research prototype and experiment artifacts are being cleaned up for release. This repository currently documents the project; runnable code, configurations, and selected trajectories will be added incrementally.

## Research questions

- Can an RLM use a REPL and external state to manage long interactive trajectories more effectively than a standard prompting loop?
- Can multi-turn post-training teach a small open-weight model to explore visual environments and make sustained progress?
- How should observations and rewards be designed so that higher training reward corresponds to actual task completion?

## Project evolution

### 1. Zero-shot RLM baseline

I first adapted an RLM-style agent to interact directly with ARC-AGI-3. The REPL let the model inspect, crop, compare, and summarize visual states without repeatedly placing every full frame in the prompt.

The baseline established a useful interaction interface, but did not reliably solve long-horizon levels. Its main failure modes were weak visual grounding, growing context, repeated ineffective actions, and dependence on the capabilities of the base model.

### 2. Multi-turn GRPO with SkyRL

I then integrated ARC-AGI-3 as a stateful SkyRL environment and built a multi-turn GRPO training pipeline. Experiments used Qwen2.5-3B and Qwen3-8B policies, with FSDP for distributed training and vLLM for rollout generation on a 4×A6000 node.

The pipeline includes:

- stateful environment interaction and action validation;
- compact adjacent-frame-diff observations;
- structured reward components and trajectory metadata;
- rollout logging and an interactive trajectory visualizer;
- oracle-distance warm-up rewards and KL regularization;
- an experimental extension to ALE-Bench as a denser-feedback testbed.

## Main findings

1. **More context did not fix weak grounding.** Supplying complete histories or repeated frames increased token usage without ensuring that the policy understood state changes.
2. **Aggregate reward was not a sufficient success metric.** Reward ablations raised scalar scores while level completion remained unchanged.
3. **Trajectory inspection exposed reward hacking.** The policy learned repetitive clicks that triggered small rewards without advancing the task.
4. **Frame differences improved observability, not necessarily reasoning.** Compact diffs made long trajectories cheaper and easier to diagnose, but the model could still ignore or misinterpret them.
5. **Oracle guidance is better used as progress supervision than as a single action script.** Oracle-distance rewards preserve multiple valid paths, though stronger initialization or a curriculum is still needed.

## Planned repository structure

```text
arc-agi-3-agent-post-training/
├── arc_agi3/          # SkyRL environment, observations, and rewards
├── rlm_baseline/      # Zero-shot RLM agent and evaluation scripts
├── scripts/           # Training and evaluation launchers
├── visualization/     # Rollout and oracle trajectory viewer
├── configs/           # Reproducible experiment configurations
└── docs/              # Experiment notes and selected case studies
```

## Next steps

- Release a minimal reproducible ARC-AGI-3 environment integration.
- Add representative training configurations and sanitized rollout traces.
- Package the trajectory visualizer and document the reward-ablation results.
- Evaluate oracle-state curricula and behavior-cloning warm-up before GRPO.

## Outcome

The trained policies did not yet achieve stable level completion. The main outcome is an end-to-end experimental system and a set of behavior-level findings about observation design, reward hacking, and credit assignment in long-horizon agent post-training.

## Acknowledgments

This project builds on [SkyRL](https://github.com/NovaSky-AI/SkyRL) and the ARC-AGI-3 environment. It is an independent research project and is not affiliated with the ARC Prize Foundation or the SkyRL authors.
