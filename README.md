# CIRA

**Composition-Informed Receiver Adaptation** for continual detector adaptation in decentralized learning.

This repository contains the current core implementation, software tests, native-protocol templates, and a paper-to-code mapping. The Python package remains `softcas` for compatibility with existing commands and records.

## Current status

The source implements complete detector states, local adapter revision, soft receiving, shared archives, parent-relative gain feedback, ridge prediction, budget admission, audit, and fallback. The source snapshot dated 2026-11-03 passed **166 CPU software tests**.

The native study is not ready to reproduce all manuscript results. External-baseline dispatch refers to a missing `softcas.external` module; donor policies and the registered half-payload interruption have not been connected to trajectory execution; native data, development locks, costs, and device profiles remain unbound. See [implementation status and paper alignment](docs/PAPER_ALIGNMENT.md).

The manuscript separately reports a numerical policy study and an independent reduced feature-module experiment, with native detector and device protocols specified as plans. This repository's native implementation is a separate execution object. Passing software tests does not establish those experimental results or device measurements. The corresponding result-generation packages are not included in this initial source snapshot.

## Install and check

Use Python 3.12 for the checked CPU environment. From a new environment:

```bash
python -m venv .venv
# Activate .venv for your operating system, then:
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[yolo]'
python -m pip install numpy==2.3.5 scipy==1.18.1
python -m unittest discover -s tests -v
python -m softcas.production --help
```

`requirements-lock.txt` records the checked core CPU versions. GPU and Jetson installation references are in [the environment guide](docs/WINDOWS_5080_ORIN.md). The checked local environment is not an independently verified clean installation on every platform.

## Source layout

| Path | Content |
| --- | --- |
| `softcas/state.py`, `model.py`, `actions.py` | Complete state, local adapters, receiving, and state construction |
| `softcas/controller.py`, `proposals.py`, `engine.py` | Candidate pool, ridge prediction, resource-aware scheduling, archive, and audit |
| `softcas/data.py`, `fitting.py`, `evaluation.py` | D/V/A/T roles, epoch accounting, and evaluation identities |
| `softcas/cache.py`, `transport.py`, `storage.py` | Exact package versions, transfers, and retained states |
| `softcas/study.py`, `conditions.py`, `study_execution.py` | Study registration, condition construction, and bound execution interfaces |
| `softcas/mechanism.py`, `decision_audit.py`, `rgb_evaluation.py` | Snapshot comparisons, same-pool decision audit, and frozen RGB readout |
| `softcas/production.py` | Explicit preparation, planning, execution, and reporting commands |
| `configs/` | Functional fixtures and unbound native configuration templates |
| `tests/` | CPU software regression tests |
| `manifests/` | Source hashes, paper-source hashes, and snapshot validation |

## Protocol templates

`configs/fixture.json` is a software-behavior fixture with `SYNTHETIC_BEHAVIOR_ONLY` and `synthetic_quality_NOT_AP` identities.

`configs/native_r1000.json` specifies four stages, 1000 fitting rounds per client, two epochs per effective fitting round, at most 500 epochs per stage, local/receiving blocks of at most 32/16 epochs, 64 attempts, 65 selection evaluations, three proposals per scheduling type, and an eight-state archive. R300 and R500 specify separate complete four-stage conditions.

These native templates retain `DEVELOPMENT`, zero unbound F/W allowances, and pending asset/calibration fields. The Q0 and Serial 50:50 settings in `study.example.json` are unlocked examples. They are not the numerical paper study's locked Q2 choice. Set native parameters from an independent native DEV lock; do not copy numerical-study settings without that selection.

The target native hardware is four Jetson Orin Nano 8GB nodes; RTX 5080 serves SOURCE preparation and offline analysis. No native detector training or device measurements were performed during this source publication.

## Scope of this source snapshot

Datasets, image content, trained weights, runtime credentials, local environments, generated outputs, and historical archives are excluded. Native execution requires externally supplied assets and the missing implementation work listed in the alignment document. Repository publication does not mark those tasks complete.
