# Decision Memo: Selecting a Constraint-Safe Liver World Model

**Decision:** recommend **RateAnchor** as the practical prototype. It has the strongest selected free-rollout ratchet MAE (`0.0207`), held-out susceptibility MAE (`0.0563`), unseen-UDCA MAE (`0.0274`), and beyond-training-horizon MAE (`0.0788`). It preserves hard clinical constraints by construction.

This is not a claim that RateAnchor learned real liver biology. The synthetic generator is both data source and quality bar, so strong performance cannot distinguish a world model from an effective generator-inverter. The evidence supports a narrower prototype-selection claim: among the tested models, RateAnchor gives the strongest observed balance of accuracy, OOD robustness, safety guarantees, and inspectable patient-specific calibration.

## Why Start With A JEPA-Style Predictive Latent?

The problem is more than next-state regression. A useful model must roll a bounded clinical state forward through hidden patient variation, treatment timing, and long horizons without invalid states or a collapsed latent. A JEPA-style predictive latent is attractive because it can represent future-relevant disease dynamics without using the eight raw fields as its complete internal state.

I treated that as a hypothesis, not an assumption. The direct supervised transformer reaches `0.0322` ratchet MAE, but has only `0.29` decompensation recall in deep evaluation. Plain JEPA does not justify itself on the primary metric: `0.0332` ratchet MAE, slightly worse than the baseline. Adaptive JEPA gives a narrower benefit: beyond-horizon MAE improves from `0.1169` to `0.1124`, and effective rank rises from `9.41` to `10.31`. The stronger latent result comes from multi-horizon, meta-adapted prediction: `0.0254` ratchet MAE, `0.0802` held-out susceptibility MAE, and decompensation recall `0.51`.

The conclusion is not "discard JEPA." Retain learned latent dynamics and patient adaptation where they help, but do not make them the entire solution. The final model keeps the successful latent/ODE recipe and adds a transparent correction for rate-shift brittleness.

## Final Architecture And The Mechanism It Tests

RateAnchor encodes observed history with a state MLP plus GRU, applies GraphRefine similarity message passing across patients in a batch, adapts a small task code from the held-in tail, and rolls the latent forward with a learned drift-only ODE. A shared constrained output head decodes the future state. The decisive addition is a rate anchor: the model measures each patient's realised positive creep in the observed ratchet fields and converts it into a per-field gain that scales constrained forecast increments. Gain `= 1` exactly recovers the base ODE head.

The continuous-time ablation makes this decision specific. The reaction-diffusion PDE performs poorly (`0.0508` ratchet MAE; `0.2087` beyond-horizon MAE). Removing diffusion while retaining the broad latent/meta recipe yields a drift-only ODE. The causal RateAnchor comparison uses the same 250k-parameter scale and the same 15-epoch budget: ODE ratchet MAE is `0.02535`, held-out susceptibility MAE is `0.11610`, and beyond-horizon MAE is `0.08517`; RateAnchor reaches `0.02070`, `0.05631`, and `0.07875`, respectively. Diffusion is harmful for this bounded, mostly one-directional state; rate calibration addresses the remaining coefficient-shift error.

## Safety Guarantees And Their Cost

Every selected model uses `ConstraintHead`, rather than relying on the loss to learn clinical rules. It bounds state fields, makes F, D, P, and M non-decreasing, permits S to step down only at a known ERCP event, and couples M accumulation to sustained F times C. Full free-rollout evaluation reports zero constraint violations for the selected runs.

The cost is intentional bias. The head cannot represent arbitrary reversible trajectories. That is appropriate under the task semantics, but would conceal a generator misspecification if those semantics were wrong. It also does not prove the learned latent factors are biologically causal.

## Collapse, Explainability, And A Concrete Review Path

Effective rank monitors latent collapse. The anti-collapse RateAnchor follow-up raises rank from `3.46` to `11.05`, but does not improve primary MAE (`0.0209` versus `0.0207`). This is why the simpler RateAnchor remains the deployment candidate: diversity is worth monitoring, but it is not by itself a reason to replace the more accurate model.

The intended explanation for a prediction of decompensation around month 30 is mechanism based: (1) observed F/D/S/P increments determine the rate-anchor gain; (2) observed A/C, treatment, and ERCP context condition the latent ODE; (3) monotone F and sustained C drive the constrained M accumulator; and (4) the decoded state crosses the decompensation threshold. This is more auditable than an isolated attention weight because each link has a defined model quantity.

I do **not** present a patient ID and numeric month-30 trace as an existing result. The package has the exact architecture, datasets, training configuration, and recorded runs, but lacks the final RateAnchor checkpoint. The first follow-up is to package that checkpoint and render the held-out patient trace with `code/inference.py`.

## Evaluation, Failure Cases, And Next Steps

All headlines are free multi-step rollouts, not teacher-forced estimates. The harness evaluates held-out trajectories, full-rollout constraint violations, held-out susceptibility, unseen treatment timing, and rollouts beyond the 60-month training horizon. It also compares with persist-last and a stochastic conditional floor. These probes could falsify the model rather than reward only in-distribution interpolation.

Failures remain visible: plain JEPA does not beat direct supervision, the PDE branch is a large negative control, and anti-collapse improves rank without winning the primary task. The deliberate scope cuts are also explicit: no multi-seed confidence interval, no no-GraphRefine ablation, and no causal-graph attention experiment. `GraphRefine` is batch similarity message passing, not liver-causal-graph attention, so batch dependence remains a deployment risk. Finally, no synthetic-only result validates a clinical deployment claim.

Next, package the winning checkpoint, produce a patient-level decompensation explanation, run the no-GraphRefine ablation, and repeat the final comparison across seeds. Until then, RateAnchor is an evidence-backed prototype, not a clinical model.
