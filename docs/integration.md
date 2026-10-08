# Policy and environment integration

The public runtime separates policy-specific proposals and execution from the WAM selector. An adapter implements three operations: `observe`, `propose` and `execute`. Observation includes frozen encoding; prediction and comparison are performed by `Selector`. The loop dispatches the selected candidate and observes the resulting environment state at the next decision.

```python
from embodiedskills.runtime import (
    Candidates,
    ObservationFeatures,
    Selector,
    run_loop,
)

class BackendSkills:
    def __init__(self, observe_fn, propose_fn, execute_fn):
        self.observe_fn = observe_fn
        self.propose_fn = propose_fn
        self.execute_fn = execute_fn

    def observe(self) -> ObservationFeatures:
        return self.observe_fn()

    def propose(self, observation) -> Candidates:
        return self.propose_fn(observation)

    def execute(self, action) -> bool:
        return self.execute_fn(action)
```

The callbacks connect to the installed policy and environment. `observe_fn` returns single-observation arrays, without a batch dimension. `propose_fn` returns five encoded candidate actions together with a tuple of their five executable native actions. `execute_fn` executes one primitive or bounded chunk and returns whether the episode has ended, including a success, failure or time limit reported by the environment. Environment success is recorded separately from this termination flag.

```python
selector = Selector("outputs/continuous/selector.pt", device="cuda:0")
skills = BackendSkills(observe_fn, propose_fn, execute_fn)
choices = run_loop(skills, selector, max_decisions=decision_budget)
```

`max_decisions` is a caller-provided budget. The default selects at every decision. To reproduce an initial-intervention protocol, pass `intervention_steps={0}`; subsequent decisions execute candidate zero. Deterministically generate the direct proposal so sampling the alternatives does not alter the direct-policy continuation.

## Continuous action policies

For a stochastic policy such as π₀.₅, independently sample five proposals from the same current observation. The sampling helper defines one paired direct seed and four alternative seeds:

```python
from embodiedskills.sampling import candidate_sampling_seeds

seeds = candidate_sampling_seeds(
    seed_base=policy_seed_base,
    environment_seed=episode_seed,
    anchor_index=decision_index,
)
```

Forward each seed through the policy's supported noise or random-key interface. Encode the portion of each candidate that will actually execute. The default continuous model consumes five chunks of eight four-dimensional control steps. If the policy returns longer chunks, take the configured executed prefix consistently for both WAM inputs and execution. Mask unused timesteps for variable-length chunks. The policy's native normalization and action-to-environment conversion belong in the adapter.

Observable state must be available at deployment. For the MetaWorld preset it is the four-dimensional end-effector/gripper summary used during collection. Preserve the camera, image orientation, control units and progress convention used to build the feature archives.

## Spatial action policies

The CLIPort helpers use native pick/place distributions. `top_spatial_local_maxima` finds distinct spatial/rotation peaks; `make_joint_hypothesis` pairs picks and places; `select_fixed_joint_hypotheses` preserves the direct hypothesis and selects diverse alternatives by native proposal probability. Those probabilities are used during proposal construction. The WAM selector consumes the resulting physical actions.

`cliport_candidate_vector` represents each action with two seven-dimensional poses plus six normalized pixel/angle values, yielding `[5,20]`. `cliport_public_height_state4` produces the mean, standard deviation, maximum and nonzero fraction of the observed height map. `cliport_fused_rgb` extracts and resizes the fused RGB image. Retain the fused-view geometry and coordinate convention used by the frozen CLIPort policy.

## Moving existing model weights

A portable selector bundle contains the three WAM states, the shared comparison head and fitted normalization buffers. Compatible local research weights can be converted explicitly:

```bash
embodiedskills convert \
  --family spatial \
  --members /path/to/member_0.pt /path/to/member_1.pt /path/to/member_2.pt \
  --head /path/to/comparison_head.pt \
  --output checkpoints/selector.pt
```

Use `--family continuous` for the recurrent-action model. Supply the three members paired with that head during training, in the same order. Legacy conversion loads trusted local PyTorch files; it strips experiment history, absolute source paths and result metadata. Normal bundle loading uses PyTorch's weights-only loader.

The repository supplies the WAM runtime, native representation helpers and callback contract. Policy servers, simulator installation, scene restoration and benchmark rollout drivers remain in their upstream integrations. This keeps the core usable across environments without bundling simulator assets or research launch scripts.
