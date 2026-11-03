# CIRA native detector implementation — 2026-11-03

The current source passed **239 CPU software tests**. All eight external methods
have independent detector implementations: DPFL, DSpodFL, DA-DPFL, DCFCL,
pFedDIL, dFLMoE, PerFedSDE-NAS and DeModCL. Bounded RTX 5080 checks cover each
on YOLOv8n, YOLO11n and YOLO26n: four replicas and two iterations per pair,
native loss/update/decoding, and prediction-preserving checkpoint reload.

Code includes persistent graphs, sparse scheduling/regrowth, coalitions,
historical banks, exchanged experts, prepared diffusion/search and module sharing.
See [mechanisms, detector choices and commands](docs/DETECTOR_BASELINES.md).
Per-method templates are in `configs/detector_baselines/`. After binding real
inputs, check with `python -m softcas.baselines.runner --spec configs/DPFL.private.json --check`.
The multistage runner is a single-host adaptation verifier; the formal executor
retains its separate qualification and physical-budget checks.

Native probes use random eight-class models, 128 input and small actual-image D/V
batches. Auxiliary labels, diffusion preparation and DeModCL costs include explicit
software fixtures. These establish executable paths, not comparative AP, SOURCE
training, native DEV, four-device runs or Orin measurements. Detector changes are
documented rather than presented as author-native implementations.

Personal configuration is blank in [local settings template](configs/local.private.example.json). Copy it to an ignored `local.private.json`; fill device addresses, login information and real asset paths locally. Remote address placeholders in release templates are empty strings. No credentials or local user paths are supplied.

Runtime interventions, exact native asset binding, DEV locking and formal execution checks remain implemented; see [execution commands](docs/EXECUTION_BINDING.md). Model weights, images, private settings and device measurements are not in this source release. Earlier entries below are historical checks, not the current verification count.

# CIRA implementation

当前论文名为 CIRA，Python 包名和 `method="softcas"` 保留旧名称以兼容已有命令与记录。当前方法依据为 `../output/CIRA_当前唯一版本_20261101/paper/`，不再用旧 SoftCAS 稿件作为新协议的依据。

2026-11-02 已修补核心执行、候选构造、检查点、统计/输出及四节点调度，并运行当前版本的 CPU 回归和短程行为检查。验证数字、源码哈希与日志见 [修补说明](docs/CIRA_REPAIR_20261102.md) 及 `artifacts/repair_20261102/verification.json`。本轮没有执行原生 YOLO 训练、GPU/Jetson 服务或设备测量。历史20261021材料保留，不能代替当前验证。

随后依据论文—代码对应审查补齐了正式上下文传播、同候选池决策审计、原生快照机制接口及 N01—N12 去重注册；当前全套 **145项测试通过**。新增真实梯度学习示例与入口见 [对应缺口修补](docs/CIRA_ALIGNMENT_REPAIR_20261102.md)。旧100项检查与零反馈示例作为历史记录保留。

## 核心实现

| 范围 | 入口 |
| --- | --- |
| 完整状态、冻结基座/外部路径、软接收、Fresh/Refine/成员修改/Import | `state.py`, `model.py`, `actions.py`, `engine.py` |
| 25维执行前特征、ridge增益、PCG64命名流、有界候选、共享档案 | `controller.py`, `proposals.py` |
| 阶段epoch额度、按裁剪长度执行、逐epoch洗牌、失败消耗记录 | `fitting.py`, `engine.py`, `data.py` |
| 精确包版本、TCP、F/E/B/W账本、持久化、审计回退 | `cache.py`, `transport.py`, `resources.py`, `storage.py` |
| 四物理节点计划、独立节点执行与完整seed合并 | `experiment.py`, `production.py` |
| 独立测量session身份、每次50预热/200计时的参考配置、重复内汇总 | `device.py`, `measurement.py` |
| 配置×上下文诊断、D/V/A/T、配对统计与表格接口 | `diagnostics.py`, `evaluation.py`, `reporting.py`, `tables.py` |

## 两种协议

`configs/fixture.json` 是旧CPU功能检查，步骤单位为batch更新，输出 `SYNTHETIC_BEHAVIOR_ONLY` / `synthetic_quality_NOT_AP`。

`configs/native_r1000.json` 显式启用当前原生协议：64次尝试、65次V评价、K=3、8状态档案、每阶段500累计epoch、本地/接收块上限32/16epoch、batch8、AdamW。R300/R500分别见 `native_r300.json` / `native_r500.json`，每阶段150/250epoch；三者都是完整四阶段独立预算条件。

原生配置尚为开发模板。未标定F/W额度置0，避免把旧fixture成本当硬件校准；成本文件、特征尺度、开发选型与实际基座/数据仍需绑定。FORMAL入口会校验确认字段、原生epoch语义及成本/设备记录。

当前硬件模板使用4台 **Jetson Orin Nano 8GB** 在线更新，RTX 5080用于SOURCE准备/离线分析。三种检测器有各自的端口模板；需在实际八类基座上检查张量形状。安装和设备服务说明另见 `docs/WINDOWS_5080_ORIN.md`，旧文档中的单机RTX逻辑客户端是另一种执行方式。

## 本地检查

在本目录使用已有CPU环境：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m softcas.production --help
.\.venv\Scripts\python.exe -m softcas.cli check-config --config configs/native_r1000.json
```

## 原生四节点入口

填写 `configs/native_r1000_matrix.example.json` 中真实地址、明确起始时刻/逐轨迹时间、真实资产和开发锁定参数后生成计划：

```powershell
python -m softcas.production plan --spec configs/native_r1000_matrix.example.json --output plan.json
```

四台各执行自己的编号；在Linux/Jetson同样使用已部署环境的Python：

```text
python -m softcas.production run-plan --spec plan.json --node-id 0
python -m softcas.production run-plan --spec plan.json --node-id 1
python -m softcas.production run-plan --spec plan.json --node-id 2
python -m softcas.production run-plan --spec plan.json --node-id 3
```

物理计划不带node-id会拒绝执行，不能在一台机器上顺序执行四个节点后称为四设备实验。每个完整seed须合并四个节点的 `trajectory.json`，`trajectory_groups`保存合并清单。报告中客户端不是独立seed。

矩阵目前实现6个内部方法和6项机制干预，三种检测器均安排消融；另有8个独立外部算法检测实现和多阶段验证入口，见上文；软件检查与正式四设备研究的资格绑定分别记录，数值代理不能算作作者算法的正式检测结果。

`study-plan` 另行登记整个 N01—N12（含300/500/1000轮）矩阵：3420条去重训练配方、5130个分析视图；RGB端点评价不新增训练。它登记条件、复用与缺项，不是 `run-plan` 的可执行任务。`study.example.json` 中 Q0/Serial 50:50 只是未锁定的示例，所有配方保持 `planned_not_run/runnable=false`。

```powershell
.\.venv\Scripts\python.exe -m softcas.production study-plan --spec configs/study.example.json --output study_plan.json
.\.venv\Scripts\python.exe examples/run_learning_demo.py --output examples_output/new_learning_demo --clients 4 --stages 2 --seed 1001
.\.venv\Scripts\python.exe -m softcas.production decision-audit --spec configs/decision_audit.example.json --output decision_audit.json
.\.venv\Scripts\python.exe -m softcas.production mechanism --spec configs/mechanism.example.json --output new_snapshot_diagnostic
```

后两条先绑定真实动作日志／快照资产；示例路径不能直接用作真实输入。机制入口分别支持 `receiving / continuation / matched_work`。原生State没有优化器缓存，因此接续对照为完整状态、重置接收logit、重新初始化本地权重，不与旧特征模型的momentum对照混称。

## 记录与范围

每次动作保留父状态、候选参数、已准入拟合长度、执行前特征、命名随机流、实际完成/尝试步数、失败成本和部署身份。独立设备测量保留每个session的原始报告；session内P95先计算，再在一个轨迹内平均，内存取最大值，不把重复计时当新增seed。

真实八类checkpoint、scene/role注册、Orin运行库与成本profile、网络地址和启动时刻需要实际输入。本次软件检查不提供这些设备或检测性能结果。历史说明见 `docs/CPU_REFERENCE_PRE_CUDA_20261021.md`。
