# CIRA execution and asset binding

The runtime code applies donor policies and the registered single half-payload interruption. It also provides a source-bound external runner and native study-binding commands. These tools do not create detector checkpoints, qualification evidence or hardware measurements from template values.

## 1. Development selection

`development-lock` reads complete native DEV federation summaries for seeds 9001, 9002 and 9003. Its native protocol uses the eight N-L01--N-L08 settings in the paper, four Serial rules and Q0--Q6. Every candidate needs all three seed records; a failed or missing candidate blocks selection. All candidates in a cell must share the declared budget identity. The example includes the actual eight parameter definitions for all six cells.

```bash
python -m softcas.production development-lock --spec configs/development_selection.example.json --output development_lock.json
```

Each record uses `role=DEV`, `study_kind=native_detector`, `metric=AP50:95`, `score_units=fraction`, `selection_metric=final_macro_seen`, `clients=4`, `stages=4`, and the actual `seen` endpoint. `family` is `local`, `serial` or `controller`; `candidate` is a Local ID, one of the four Serial rules, or Q0--Q6. Actual `F_total` and `network_tx_bytes` are required. Local ties use lower computation, communication and ID. Global controller ties use lower computation and ID. Serial uses the selected controller's `beta_lr`, and then compares its four rules per cell. T results cannot enter this selection. A separate software-fixture scope cannot unlock native formal execution.

The output's `selection.sha256` identifies the canonical DEV evidence manifest; it is not the hash of the lock containing itself. The binding file separately pins the actual lock file's SHA-256.

## 2. Condition construction

```bash
python -m softcas.production conditions --spec configs/conditions.example.json --output new_condition_directory
```

This reads a scene/role registry and contact trace, materializes scene skew, nested D subsets, domain order and contact visibility, and writes exact bindings. It never edits raw images. Empty and weak donor policies are applied to catalogue visibility, cache admission/access, and inherited states. A weak donor policy requires a DEV-locked list of exact package hashes.

A `single_half_payload` event names receiver, owner, stage two and one actual online contact offset. The first eligible payload is cut at `floor(payload_bytes/2)`. Control frames are transmitted normally. Sender and receiver charge the actual partial bytes; the partial package is not admitted to cache. A retry starts at zero and is not interrupted again. The event claim is persisted before payload transmission and the actual sent bytes afterward. A process crash between those records remains auditable/uncertain and is not silently replayed as another injected interruption.

## 3. Native binding and plan

```bash
python -m softcas.production study-plan --spec configs/study.native.example.json --output study_registry.json
python -m softcas.production study-bindings --spec configs/study_binding_request.example.json --output binding_skeleton.json
python -m softcas.production study-bind --spec configs/native_binding_inventory.example.json --output binding_summary.json
```

Use the actual DEV lock when registering the native study. Q0 in the example is an unlocked illustration, not a selected winner. The inventory supplies actual config, base checkpoint, detector metadata, calibrated budgets, optional DEV feature scales, condition manifests and external adapters. `study-bind` hashes these files and reports each missing inventory field. With `binding_output_directory` it saves `bindings.json`, `compiled_plan.json` and `binding_report.json` to a new/empty directory.

Formal native metadata checks include the eight-class vocabulary, six declared ports, input size 640, verified scene-role separation, D/V/A support in every client-stage, independent T pools, validated cost coverage, exact frozen-base identity, and matching runtime family/device/precision. Actual tensor shapes and hardware are checked again by the native execution backend. Feature-profile schema is `cira.feature_scales.v1`, with the matching `cell`, `role=DEV`, and `features={fields:[25 ordered names], scales:{...}}`.

The existing formal confirmations remain required. The binder does not auto-set confirmation flags, invent a calibrated F/W allowance, or change DEVELOPMENT to FORMAL merely to make a plan runnable.

For four physical nodes, provide real endpoints, a shared explicit epoch and either positive per-trajectory spacing or a complete start-epoch mapping. Each node runs only its own jobs:

```bash
python -m softcas.production study-run --spec bound/compiled_plan.json --node-id 0
```

Run separately for nodes 1, 2 and 3, then use the existing `join-nodes` command. Nodes are not additional independent seeds. Every compiled job pins its fields and file identities and is checked again before execution.

## 4. External algorithms

The eight method identities are registered in `configs/external_baselines.json`. Registry entries are not eight completed detector ports. A usable external port must preserve its own algorithm, loss/decoding, historical access, topology, state updates and resource accounting. It is never routed through CIRA's internal method switch.

First snapshot an actual upstream/ported source checkout without running it:

```bash
python -m softcas.production external-source --spec configs/external_source.example.json --output external_source.json
python -m softcas.production external-bind --spec configs/external_binding.example.json --output external_binding_report.json
```

The source snapshot hashes executable code and records a full Git commit. `external-bind` additionally binds the detector runner and qualification report. The native inventory and external commands accept either its full output report or its `adapter` member. The qualification identity is `source_identity` from the source snapshot. It must be updated whenever the detector port changes.

`external-check` validates a manifest without loading third-party code or starting a trainer. `external-run` explicitly starts the declared executable argv with `shell=False`. Both `{job}` and `{result}` must be consumed by the runner. A shell wrapper or CIRA executor alias is rejected. Source files, runner scripts, qualification evidence, and native inputs are rechecked before use. Formal execution requires an actual matching Git checkout and a qualified detector port for the requested dataset/detector.

### Runner input and output

The runner receives a `cira.external_job.v1` JSON file containing the original job, protocol settings, exact input manifest, condition policy, source qualification identity and output paths. It writes `cira.external_trajectory.v1` JSON with:

- `method`, `seed`, `cell`, `study_trajectory_id` and `study_recipe` matching the input.
- `input_manifest` echoed exactly and `interventions_sha256` equal to the canonical normalized donor/contact policy hash.
- `complete` as a Boolean; incomplete runs keep endpoint scores null.
- For complete native results, `metric=AP50:95`, `score_units=fraction`, and a SHA-256-bound `matrix` file in this run's output directory.
- Matrix rows contain `seed`, `client`, `stage`, `current_domain`, `domain`, `deployed` and `score`. Formal runs cover all four stages; one deployed checkpoint identity is used across the historical domains in a client-stage. The driver recomputes SeenAP/CurrentAP/Forget/BWT from this matrix and rejects conflicting supplied summaries.
- Formal `deployed_artifacts` maps each deployed SHA-256 to its actual checkpoint file binding in this run's output directory. A checkpoint name alone is insufficient.
- Actual `F_total`, `network_tx_bytes`, `selection_E`, and `max_client_update_seconds`, with complete-accounting flags for complete formal results.

Unqualified CPU process fixtures must use `test_quality_NOT_AP`; this label cannot be promoted into formal detector AP by the execution wrapper. Timeouts, nonzero exits and invalid outputs preserve a failed record and null scores. Interrupted or failed execution requires cost/state reconciliation before retry; completed resume verifies the old result hash and input/source identity.

For physical plans the external runner manages its complete federation according to the qualified algorithm's topology. `external_coordinator_id` explicitly selects the node that launches it; the same federation is not launched once per peer. Its outputs are listed separately from internal node-join groups.

## Remaining external inputs and qualification

Required real artifacts are the eight-class base checkpoints, verified scene/role registries, native DEV records, feature scales, operator/device cost profiles, actual node endpoints/schedule, and the external algorithms' implemented detector ports and qualification evidence. The checked-in examples intentionally remain unbound. No native experiments or device measurements are claimed by software regression.
