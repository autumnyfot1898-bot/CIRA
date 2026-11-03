# Native detector baseline implementation status

Checked on 2026-11-03. Implemented update operations, native interface checks,
and formal experiment qualification are separate. A round core is not a complete
continual-study runner. Hardware addresses, login names, keys and data paths are
user configuration, left blank in release templates.

## Implemented operations

`softcas.baseline_algorithms` is independent of CIRA's State, Assembly, gain
predictor and archive. `softcas.detector_baselines` supplies native full-detector
loss, local fitting, subset evaluation, parameter updates and prediction decoding.

| Method | Implemented and checked | Still required for a complete qualified detector port |
|---|---|---|
| DPFL | GGC randomized add/remove subset selection; separate non-monotone greedy variant; receiver-V native loss oracle; sample-weighted floating-state averaging; native local SGD and personalized round | Batched BGGC preprocessing with bounded model retention/transfers; persistent graph; complete stage/history/contact/budget runner and qualification |
| DSpodFL | Synchronous ATC/CTA cores; explicit gradient/edge indicators; fixed-graph Metropolis weights; immutable snapshots; native ATC loss/update/decode | Registered indicator process and full study execution, stage histories, physical transmission, budget accounting and qualification; delayed-gradient variant not implemented |
| DA-DPFL | Active-mask coordinate normalization and receiver projection; max-wait schedule; layer-scope PQ pruning | Sparse initialization/training, trigger histories, complete scheduling, removal/regrowth and final-mask semantics; native qualification |
| DCFCL | Gradient cosine plus weighted model cosine | Coalition payoffs, merge-blocking/evolution, continual training, historical knowledge transfer and qualification |
| pFedDIL | Paper/source catalogue | Shared auxiliary classifiers, knowledge matching, historical model inheritance, weighted knowledge migration and detection qualification |
| dFLMoE | Paper/source catalogue | Head exchange, feature transforms, cross-attention expert fusion, both training phases and YOLO decoding |
| PerFedSDE-NAS | Paper/source catalogue | Detector supernet/encoding, CATE/score preparation, conditional diffusion, RBF prediction, archive optimization and preparation accounting |
| DeModCL | Verified publication identity/abstract; no verified author implementation/full technical specification | Full method specification, MTDR, task-anchor capability signatures, selective collaboration and detector qualification |

No method is automatically added to `qualified_native_detector_adapters`.
The formal executor retains its existing evidence and asset checks.

## Algorithm definitions and detector choices

- DSpodFL uses `1/(1+max(deg(i),deg(j)))` on active links. Differences use the
  pre-step model. ATC computes gradients before mixing; CTA after mixing. Only
  parameters change; gradient-copy BN buffers are discarded, as in the selected
  author functions. No optimizer momentum is introduced. Realized indicators
  are inputs, never CIRA action choices.
- DPFL implements Appendix-B GGC add/remove marginals. Equal positive marginals
  use 1/2 from `a/(a+b)`; the checked author Python leaves `p` undefined there.
  Equal zero marginals use `p=1`, as explained in the paper. Cardinality includes
  the receiver. `client_weights` specifies sample weights; the probe uses equal
  weights. Receiver V reward is `(10 - detection_loss)/10`, preserving the
  checked author implementation's negative-loss ordering. Test AP is not used.
  Floating states are averaged; integer BN counters remain receiver-local, an
  explicit detector-port choice.
- DA-DPFL helpers cover layer-scope PQ pruning. Tied pivot magnitudes remain
  active. They do not substitute for the complete method.
- Native models use eight classes and Ultralytics 8.4.0 detection losses. The
  fixed DFL projection remains frozen. No CIRA adapter is inserted.

## Actual verification

RTX 5080 / PyTorch 2.7.1+cu128 / Ultralytics 8.4.0 were used for bounded checks
of both round cores on YOLOv8n, YOLO11n and YOLO26n at 640 input. Each uses four
independent replicas and eight actual ACDC train images/official boxes: one fit
and one distinct selection image per weather. Six method/architecture pairs and
24 client updates were checked, including finite decoding and checkpoint reload.
These random-initialization checks are not SOURCE pretraining, DEV results,
formal AP, four-device runs or Orin measurements.

```bash
python -m softcas.detector_baselines --manifest /absolute/probe_manifest.json --output /new/output --architecture YOLOv8n --device cuda:0 --image-size 640
```

The manifest declares `scope=native_interface_probe_NOT_FORMAL`, four `clients`,
and each client's nonempty `fit` and `select` records. Each record has `image`,
its `sha256`, and normalized eight-class YOLO `boxes`. Manifest/source hashes are
recorded. Nonempty outputs are not overwritten; failures retain reports. Images,
labels and checkpoints are not distributed in this source release.

Selected upstream comparisons checked one DSpodFL ATC snapshot case, 25 seeded
DPFL subset choices, nine DA-DPFL wait schedules and ten PQ-pruning cases. These
are kernel checks, not full author experiment reproductions.

## Primary sources

- [DPFL paper](https://proceedings.mlr.press/v258/kharrat25a.html), [author code](https://github.com/salmakh1/DPFL/tree/000a96827744010cb440b8f990c48841a25d0014).
- [DSpodFL paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html), [author code](https://github.com/ShahryarBQ/DSpodFL/tree/1f1511b597ac878f9d7d98367b0aaae2a5a0be24).
- [DA-DPFL code](https://github.com/EricLoong/da-dpfl/tree/a7f93b8b9eb1adc75aa43328f39a9a83e3a854f5), MIT.
- [DCFCL code](https://github.com/ydn3229/DCFCL/tree/23e94ea285f403edd68e493796959fdc16058a95), MIT.
- [pFedDIL paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06262.pdf).
- [dFLMoE paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xie_dFLMoE_Decentralized_Federated_Learning_via_Mixture_of_Experts_for_Medical_CVPR_2025_paper.pdf).
- [PerFedSDE-NAS paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Liao_Neural_Architecture_Search_Driven_by_Locally_Guided_Diffusion_for_Personalized_ICCV_2025_paper.pdf).
- [DeModCL publication](https://doi.org/10.1109/TMC.2026.3726945).

Upstream files are not redistributed. Local snapshots were used for inspection
and selected-function comparisons only.
