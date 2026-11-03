# Eight detector baseline implementations

Updated 2026-11-03. All eight methods have independent implementations in
`softcas/baselines/`, factory entries, method settings, persistence and a common
multistage verification runner. They do not call CIRA's action selector, state
archive or gain controller. These are documented detector adaptations of the
cited algorithms, rather than the authors' original classifier programs.

## Implemented mechanisms

| Method | Implemented mechanisms | Entry |
|---|---|---|
| DPFL | Local pretraining; streamed two-pass BGGC with bounded peer retention; persistent graph; receiver-V GGC; sample-weighted personalized averaging | `dpfl.py` |
| DSpodFL | Synchronous ATC/CTA; seeded gradient/edge Bernoulli indicators; fixed-graph Metropolis mixing; pre-step snapshots and event-process restoration | `dspodfl.py` |
| DA-DPFL | ERK/uniform sparse initialization; same-round reuse and maximum wait; mask-normalized aggregation; norm-history trigger; layer-scope PQ pruning; cosine removal and RigL regrowth from full-D averaged gradients | `dadpfl.py` |
| DCFCL | Historical-teacher logit distillation; gradient/model coalition utilities; MBA merge-blocking and dynamic evolution; partition inheritance; blocking-coalition certificate | `dcfcl.py` |
| pFedDIL | Model/auxiliary portfolios; domain matching; historical inheritance; correlation-weighted migration; server aggregation; per-image prediction fusion | `pfeddil.py` |
| dFLMoE | Local and frozen exchanged Detect heads; feature transforms; attention fusion; two training/freezing phases; native dense decoding | `dflmoe.py` |
| PerFedSDE-NAS | Detector supernet; architecture/data encoders; score preparation and guided diffusion; RBF screening; archive training; nonzero-update server aggregation | `perfedsde.py`, `nas_*.py` |
| DeModCL | MTDR memory traces/feasibility; MSA anchors/alignment/capability signatures; CMD selective sharing; cached signatures; latency/storage/loss constraints; module-pool inference | `demodcl.py` |

Each saves its method-specific graph, RNG, masks/history, coalition partition,
model bank, experts/router, search population/generator or module/signature pool.
The runner also saves PyTorch/CUDA RNG and seals resume to source and spec hashes.

## Detector adaptation choices

- Native losses use Ultralytics 8.4.0 and eight fixed detection classes. YOLO's
  DFL projection stays frozen. Floating-state averaging keeps integer counters
  receiver-local (server-local for NAS). No CIRA adapter is inserted in the first
  six methods.
- **DPFL:** graph cardinality includes the receiver. Equal positive add/remove
  marginals use probability 1/2; inspected author Python leaves this case
  undefined. Equal zero marginals use 1 as in the paper. The selection oracle is
  negative receiver-V loss, never T AP. The fixed graph survives domain changes.
- **DSpodFL:** the selected variant is synchronous ATC/CTA; optional delayed-
  gradient asynchronous experiments are outside this variant. Metropolis weights
  use fixed-graph degrees. Gradient-copy BN buffers are discarded as in the
  selected author update functions.
- **DA-DPFL:** layer-scope PQ pruning is selected. Regrowth uses squared gradients
  averaged over D batches, freezing BN for importance evaluation. Final deployed
  weights are projected onto the current mask; this convention is explicit.
- **DCFCL:** Eq. 11-12 use separate gradient and model cosines rather than the
  inspected author's combined-affinity shortcut. The paper's MBA output is kept.
  Some utilities retain blocking coalitions: traces report
  `blocking_equilibrium_verified=false` and list them. A negative certificate is
  covered by a regression test. Native pre-decoding class logits supply KD, with
  the student's detection forward reused to avoid duplicate BN updates.
- **pFedDIL:** the coordinating server is retained. Historical auxiliary heads
  share detector features. D records require explicit binary `auxiliary_target`
  membership labels; object classes are not substituted. The paper does not fully
  specify negative sampling, so its D-only construction must be caller supplied
  and recorded. Dense predictions fuse on aligned native anchor grids before
  common decoding.
- **dFLMoE:** compatible Detect heads are exchanged within each architecture.
  Feature transforms and attention fuse regression/class logits at the three
  native head levels. Phase two freezes backbone/all heads while gradients flow
  through the heads into transforms/router. YOLO26 one-to-one and one-to-many
  paths are handled separately. Cross-architecture head copying is not performed.
- **PerFedSDE-NAS:** the original server is retained. The detector search space
  has six residual bottleneck sites and a global ratio bit (1/16 or 1/8), with at
  least one active site. An operation Transformer uses paired masked
  reconstruction in the style of CATE; a directed forward/backward GCN encodes
  architectures. These are detector-space reimplementations, not author
  CATE/DiGCN checkpoints or original classification search spaces. RBF screens
  generated candidates using a truly evaluated population. Within-stage fitness
  is inherited; changing domain clears old fitness and requires current-stage D
  descriptors. Offline SOURCE/DEV preparation costs remain separate. Fixture
  preparation cannot pass the native asset loader.
- **DeModCL:** the supplied primary full text defines Eq. 18-37 and Algorithm 1.
  Here a module is a complete compatible detector, projection and memory trace.
  Raw D images are not shared. Trace statistics use Welford updates and normalized
  prototypes. Assignment threshold, radius update and histogram bins are explicit
  port choices where the paper is underspecified. Anchor histograms use
  normalized feature activations and a shared fixed random projection. Transfers
  are versioned and incompatible parameter copying is rejected. C1/C2 admit
  actions; C3 suppresses invalid results/signature publication. No eviction is
  added. A `round` is a task-arrival slot, not a federated synchronization round.
  Real cost estimates must be bound by the caller.

## Commands and inputs

Copy one of `configs/detector_baselines/*.example.json` to an ignored
`*.private.json`. Bind real checkpoint/registry paths and SHA-256, output, device
and iteration count. Example hyperparameters are not DEV selections. Algorithm
communication iterations/task slots do not imply CIRA fitting rounds or epochs.

```bash
python -m softcas.baselines.runner --spec configs/DPFL.private.json --check
python -m softcas.baselines.runner --spec configs/DPFL.private.json
python -m softcas.baselines.runner --spec configs/DPFL.private.json --resume
```

The runner checks image hashes and vocabulary, fits on D, selects on V, records
independent A diagnostics and evaluates current/previous domains on T. Outputs
include per-client artifacts, historical-domain AP matrix, diagnostics, failed
attempts and restart checkpoints. AP50:95 uses fractions; percentage-point
differences require x100. Completed resume does not repeat fitting/matrix rows.

The registry follows `softcas/data.py`: its document has `labels` and `records`;
records identify client, stage, domain, role, scene_id, image, image_sha256 and
normalized YOLO boxes. Preserve disjoint scenes across D/V/A/T. Personal paths,
device addresses, logins and keys remain blank in released templates.

### NAS preparation

```bash
python -m softcas.baselines.prepare --spec configs/diffusion_preparation.private.json
```

Bind a real torch artifact: schema `cira.native_architecture_triples.v1`, scope
`native_detector_preparation`, metric `AP50:95`, score_units `fraction`,
architecture one-hot tensors, dataset features, measured performances,
SOURCE/DEV roles, adjacency, paired architectures and paired masks. Set device
and iterations. Three explicit training phases produce the score/meta artifact
and input hashes; no author weights or AP triples are automatically invented.

`nas_descriptor.region_descriptor` builds D-only labelled-ROI descriptors with a
bound ResNet-18 encoder, ImageNet normalization and 512 features. Absent classes
have masks; repeated crops do not increase independent sample count. Save schema
`cira.nas_dataset_descriptors.v1`, role `D`, `descriptors` (one per client), and
`by_stage` mapping string stage IDs to current descriptors. Each descriptor can
contain `values` (classes x samples x 512) and `valid` masks. Bind this and the
prepared diffusion artifact by path/hash. Later stages require their own D
descriptors, rather than silently reusing the first domain.

## Verification and execution scope

The release passes **239 CPU software tests**. Bounded RTX 5080 checks cover
**8 methods x 3 YOLO architectures**, four independent model replicas per pair,
two iterations, native D loss/gradients, finite decoding, checkpoint reload and
equal predictions after reload: 96 replicas and 192 client-iterations, not 96
independent performance seeds. Inputs are random eight-class models, 128-pixel
images and eight actual ACDC train images/official boxes (one D and one distinct
V image per weather). Positive-only auxiliary labels, tiny numerical diffusion
preparation and numerical DeModCL costs are explicit software fixtures. They do
not validate comparative AP, task matching, prepared search accuracy or costs.

```bash
python -m examples.verify_detector_ports --manifest /absolute/probe_manifest.json --output /new/output --architecture YOLOv8n --device cuda:0 --clients 4
```

The manifest requires `scope=native_interface_probe_NOT_FORMAL`; client `fit` and
`select` records contain image, sha256 and normalized boxes. Source hashes and
sanitized summaries are included; raw images/labels/checkpoints are excluded.

The multistage runner is a single-host logical federation with
`NATIVE_ADAPTATION_VERIFICATION` scope. It does not implement physical wire-byte
accounting: network bytes are null; FLOP diagnostics retain unknown operators.
Its A score is diagnostic, not an invented common release rule. Formal
300/500/1000 fitting-round studies, matched budgets and four-device qualification
continue through `softcas.external` and execution-binding checks.
`qualified_native_detector_adapters` stays empty until real-study qualification.
Implemented code/probes do not create SOURCE weights, native DEV, formal AP or
Orin measurements.

## Primary sources

- [DPFL paper](https://proceedings.mlr.press/v258/kharrat25a.html), [pinned author code](https://github.com/salmakh1/DPFL/tree/000a96827744010cb440b8f990c48841a25d0014).
- [DSpodFL paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html), [pinned author code](https://github.com/ShahryarBQ/DSpodFL/tree/1f1511b597ac878f9d7d98367b0aaae2a5a0be24).
- [DA-DPFL code](https://github.com/EricLoong/da-dpfl/tree/a7f93b8b9eb1adc75aa43328f39a9a83e3a854f5), MIT.
- [DCFCL paper](https://arxiv.org/abs/2509.23683), [pinned author code](https://github.com/ydn3229/DCFCL/tree/23e94ea285f403edd68e493796959fdc16058a95), MIT.
- [pFedDIL paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06262.pdf).
- [dFLMoE paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_dFLMoE_Decentralized_Federated_Learning_via_Mixture_of_Experts_for_Medical_CVPR_2025_paper.pdf).
- [PerFedSDE-NAS paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liao_Neural_Architecture_Search_Driven_by_Locally_Guided_Diffusion_for_Personalized_ICCV_2025_paper.pdf).
- [DeModCL](https://doi.org/10.1109/TMC.2026.3726945), supplied primary PDF SHA-256 `279ce1184d5da57c0f28e4448b869f23df6acea384dc769d9a87faf512eda367`.

Upstream snapshots were inspected; selected-kernel equivalence checks from the
previous release are retained. Neither upstream sources nor full-text papers
are redistributed. The adaptation choices above separate detector implementations
from original classification experiments.
