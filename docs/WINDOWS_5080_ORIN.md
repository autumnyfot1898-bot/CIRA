# Windows RTX 5080 与 Jetson Orin

2026-10-21：本轮仅编写和静态阅读源文件，未执行下面的命令。Jetson 按 Orin 标准版适配，具体 SKU、RAM、JetPack 留在配置中。

## 环境

`bootstrap_5080.ps1` 创建 `.venv-5080`，采用 torch 2.7.1 / torchvision 0.22.1 CUDA 12.8 包和 Ultralytics 8.4.0。选择依据是 [PyTorch 2.7 Blackwell/cu128 公告](https://pytorch.org/blog/pytorch-2-7/) 和 [官方安装版本表](https://pytorch.org/get-started/previous-versions/)。显式运行时检查实际 GPU、计算能力、wheel 架构支持，不能静默切 CPU。

`bootstrap_jetson.sh` 要求 Python >=3.10，复用 JetPack GPU 环境，或接收 `TORCH_INSTALL`、`TORCHVISION_INSTALL` 指定的 NVIDIA 包。匹配关系参见 [Jetson PyTorch 安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/) 和 [NVIDIA 版本表](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)。桌面 cu128 wheel 不用于 Orin。

TensorRT 在目标 Orin 上创建，绑定精确硬件/软件/精度/形状/权重身份，兼容 10 tensor-address/v3 和 8 bindings/v2 API，参考 [Python API](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html) 和 [迁移说明](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-python-api.html)。默认 `executor: torch`；`tensorrt_adapters` 替换残差分支，保留显式组合，软接收不会按系数跳过分支。

## 数据和基座

命令以项目根目录为工作目录。JSON 相对路径通常相对工作目录；registry 内相对图像路径相对 registry 目录；`extends` 相对配置文件目录。

`prepare-data` 的 COCO spec 包含 `annotation_files`、`image_roots`、`scene_map`、`output`，前两项为相同域键的映射。Scene map 使用 `域/图像文件名` 为键，显式声明场景、角色及核验标志。Cityscapes 使用 `format: cityscapes_polygons`、`appearance_scene_map` 和 `output`；appearance map 是记录列表，每条包含 `image`、`polygon_annotation`、`scene_id`、`role`、`domain`、`verified_grouping`，可给 `client`、`stage`。按可见 polygon 包络转换框，记录坐标约定，不称为官方检测框。

八类顺序固定为 person、rider、car、truck、bus、train、motorcycle、bicycle。Registry 记录包含 `image`、`image_sha256`、`scene_id`、`role`、`domain`、`client`、`stage`、`boxes`、`verified_grouping`；框格式为 `[class,cx,cy,w,h]` 原图归一化坐标，允许空标注。Whole-scene 分区先固定全局角色及唯一在线 appearance，再用共享 targets 分配客户端，不按检测分数挑样本。

`prepare-base` 仅接收 SOURCE/DEV；图像副本和 YOLO 标签写到输出目录，原图旁不写标签。SoftCAS 在线更新冻结基座/检测头/BN/缓冲区，只训练允许的 local phi/beta。三个 detector compatibility groups 的基座与端口分别配置。

使用 `relocate-registry` 迁移 Windows → Orin 图像路径；内容哈希、场景、角色、标注及排序身份保持一致。Agent 的 `bases` 为 `{精确checkpoint_SHA256: /目标文件.pt}`，`registries` 为 `{registry_id: /目标registry.json}`。目标预放 A/T 数据，RPC 只传状态/组件/控制信息。

## 以后显式执行的命令

先填写真实配置。以下命令本轮均未执行。

```powershell
# Windows PowerShell，工作目录 softcas_impl
.\bootstrap_5080.ps1
$env:SOFTCAS_PEER_TOKEN = '<两端一致的实际令牌>'
.\.venv-5080\Scripts\python.exe -m softcas.production partition --spec configs/partition.example.json
.\.venv-5080\Scripts\python.exe -m softcas.production prepare-base --spec configs/base_preparation.example.json
.\.venv-5080\Scripts\python.exe -m softcas.production calibrate --spec configs/calibration.example.json
.\.venv-5080\Scripts\python.exe -m softcas.production run --spec configs/job.example.json
.\.venv-5080\Scripts\python.exe -m softcas.production plan --spec configs/matrix.example.json --output runs/plan.json
.\.venv-5080\Scripts\python.exe -m softcas.production run-plan --spec runs/plan.json
```

```bash
# Orin，先放基座和本地 registry，再启动 agent
bash bootstrap_jetson.sh
export SOFTCAS_PEER_TOKEN='<两端一致的实际令牌>'
.venv-jetson/bin/python -m softcas.production serve-agent --spec configs/jetson_orin.json
# 另一终端，对已提交部署做图片推理
.venv-jetson/bin/python -m softcas.production predict --spec configs/predict_orin.example.json
```

`doctor` 只在以后显式调用时检查环境。`diagnostic` 读取两个结构、固定 h/beta、开发记录和因果 donor，四格拟合后先锁定 V 选择再读 T。`report` 输入包含 `trajectory_files`、`output`；正式主表拒绝开发/合成 AP，缺失种子不补零。

## 网络、审计、部署

每个客户端独立 cache/archive/source 服务/额度/journal。Contact trace 采用共同 epoch 的相对秒数，只允许观察当时在线对端的广告，离线完整已校验包仍可用。物理分布配置必须填写数值 `epoch_start_unix`，不能保留 null。源端 payload 串行，部分传输/重试及控制字节均计费，重启从零发送。

搜索只使用本阶段 D/V；E 计唯一完成的完整 V 评价。失败返回 parent，target=0，已发生的费用及缓存变化保留。锁定一次 champion 后，Orin 一个 worker 做一次本地 A 比较，另一个只构建候选的 worker 做设备检查。失败保留 incumbent，不重新搜索。Receipt 和 compare-and-swap 原子指针保证只激活锁定且合格的候选；提交 ACK 丢失先查 status，无法确定则输出 deployment unresolved 并停止下一阶段。

部署后，独立 T 检查所有已见域；另起进程测量实际 deployed 身份，避免把被拒 champion 的指标当作最终结果。延迟包括预处理 CPU tensor→H2D→检测器→固定 NMS→D2H，不含文件读取/解码。RSS 为隔离 worker 进程峰值，包含构建期；CUDA allocated/reserved、统一内存系统占用另列，不相加。

Agent 配置有 bundle 暂存、日志、job_store、engine_store 额度。存储耗尽拒绝新任务，保留已部署包、receipt 和失败证据；watchdog 周期检查 RAM/时间/磁盘，检查间隔或单算子内可能超额，原样记录。源端已发送 payload 配额从持久传输记录恢复。

`--resume` 校验完整输入/配置/实现哈希，仅复用完整阶段。中断 journal 不会自动重跑搜索或 A；先用 `target-status` 核对目标，保留原记录。旧 CPU 参考入口仍在 `softcas.cli`，当前硬件入口是 `softcas.production`。

## 成本、实验和输出

ATen 算术和控制器线性代数采用 MAC=2 的标量运算约定，不声称等同 GPU issued instructions。未知算子明确标记，strict 模式停止。Profile 绑定 GPU/软件/精度、基座和端口；种子不属于硬件身份。用于组件标定的人工外部分支只是算术样本，不发布为真实 donor 或性能证据。Admission 用非负组件回归和开发残差包络，是估算器，不证明未见组合硬上界。

TensorRT builder autotuning 不在可见 ATen 中；严格流程需预先准备目标引擎，登记精确 plan 每层成本。缺失时不输出完整 F 结论。默认原生 PyTorch 完成方法，不强制 TensorRT。

矩阵示例两数据集×三检测器×六方法×五种子，共 180 条主轨迹；主检测器六消融另有 60 条。一个 seed 包括全部客户端/阶段，客户端/设备重复不增加统计 n。多物理节点用 `join-nodes` 后才组成完整 seed。敏感性任务通过单任务覆盖配置改变预算/接触/客户端/分区并使用独立目录，不混入主矩阵。

每阶段输出 `journal.jsonl`、`actions.jsonl`、`ledger_live.jsonl`、`stage.json`、checkpoints、packages、传输日志、`independent_test.json`、`deployed_device.json`。根目录有 `progress.json`、`test_matrix.jsonl`、`deployed_endpoints.csv`、`trajectory.json`、`all_transfers.json`、源端账本。设备汇总采用最终各客户端部署的最大值并保留逐客户端结果；F/网络记录不完整时汇总为 null，另存已知下界。
