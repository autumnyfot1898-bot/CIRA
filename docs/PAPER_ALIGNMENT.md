# Paper-to-code alignment — 2026-11-03

Reference: the current CIRA manuscript and supplement, identified by `manifests/PAPER_SOURCE_SHA256.json`. Manuscript content and results were not changed for this source publication.

## Checked core correspondences

| Manuscript definition | Implementation |
| --- | --- |
| Complete state `(z,h,C,phi,beta,nu)` | `softcas/state.py`; optimizer moments are not inherited across fitting blocks |
| Frozen external adapters; null branch and softmax receiving coefficients | `softcas/model.py` |
| Fresh/Refine/membership revision/Import | `softcas/actions.py`, `softcas/proposals.py`, `softcas/engine.py` |
| Signed child-minus-parent validation feedback; failed target zero | `softcas/engine.py` |
| Ridge sufficient statistics, linear solve, feature coverage, parent-to-champion gap, resource-normalized priority | `softcas/controller.py` |
| Incumbent/champion protection and bounded archive | `softcas/controller.py` |
| R1000 epoch limits, 64 attempts, 65 V evaluations, 3 proposals/type, 8 archive states | `configs/native_r1000.json` |
| Roles, paired summaries, failures and missing records | `softcas/data.py`, `evaluation.py`, `reporting.py`, `study.py` |

These are checked source correspondences, rather than a proof of complete end-to-end equivalence.

## Known implementation and binding gaps

1. `production.py` and `study_execution.py` import `softcas.external`, which does not exist in this snapshot. The external-baseline commands and dispatch path cannot execute; the eight external native detector adapters and their qualification have not been implemented.
2. `conditions.py` generates donor policies and a registered single half-payload intervention. `study_execution.py` copies them to the job's `interventions` field, but `execute_trajectory` does not consume that field. Empty/weak-donor policies and the registered interruption are therefore not fully applied by the native execution path. Half visibility is different: it is materialized in the contact trace.
3. Native data/checkpoint identities, development-selected settings, feature scales, compute/time allowances, device profiles, addresses and causal start schedules remain external unbound inputs. The templates intentionally prevent formal execution without them.
4. The paper's numerical policy study and independent reduced feature-module experiment use different execution objects. Their complete result-generation packages and data are not part of this initial repository snapshot. Current `softcas_impl` should not be described as the executor that generated every reported table.

## Verification

The publication staging copy passed the full CPU software regression. Exact test count/result and source identities are in `manifests/SNAPSHOT_VALIDATION.json` and `manifests/SOURCE_SHA256.json`. The source, tests and configuration files were copied byte for byte from the local CIRA implementation; repository-facing documentation was prepared separately.

The check did not perform native YOLO training, launch the four Jetson services, or measure devices. No experimental scores were added or inferred from software tests.
