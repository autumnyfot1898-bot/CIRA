"""Small, executable CIRA walkthrough. CPU toy data only; never AP evidence."""
from pathlib import Path
import argparse
import csv
import hashlib
import html
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CIRA运行示例</title>
<style>
:root{color-scheme:light;--ink:#263e4b;--muted:#657985;--line:#dbe5eb;--blue:#537ea8;--green:#4a9c90;--orange:#d08a60;--purple:#8f7eaa}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1220px;margin:32px auto;padding:0 22px 70px}h1{font-size:32px;line-height:1.4;margin:6px 0 12px}h2{font-size:20px;margin:0 0 15px}.scope{padding:10px 16px;background:#e9f2f5;border-left:3px solid var(--blue);font-size:14px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:22px 0}.box{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px}.cards .box{margin:0}.big{font-size:30px;font-weight:650}.muted{color:var(--muted);font-size:14px}.grid{display:grid;grid-template-columns:1.4fr 1fr;gap:18px}.controls{display:flex;gap:12px;align-items:center;margin-bottom:18px;flex-wrap:wrap}select,button{font:inherit;padding:7px 12px;background:white;border:1px solid #c4d3dd;border-radius:6px}a{color:#236d90}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid var(--line);white-space:nowrap}th{font-weight:600;color:#5a7484}.scroll{overflow-x:auto}.flow{display:flex;gap:10px;overflow-x:auto;padding:15px 0}.step{min-width:105px;padding:12px;border-top:4px solid var(--blue);border-radius:7px;background:#f3f7fa;cursor:pointer}.step.active{outline:2px solid #405f76;outline-offset:2px}.step small{display:block;color:#647a87}.steps-label{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:#5c7280}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.detail{background:#f2f6f8;border-radius:8px;padding:15px;overflow-wrap:anywhere;font-size:14px}.detail code{font-size:12px}.bar{height:12px;background:#e7eef2;border-radius:8px;overflow:hidden;margin:10px 0}.fill{background:var(--green);height:100%}pre{white-space:pre-wrap;word-break:break-all;font-size:13px;line-height:1.8}svg{width:100%;height:230px}svg text{font-family:inherit}footer{margin-top:24px;color:var(--muted);font-size:14px}.pill{display:inline-block;background:#edf4f7;border-radius:20px;padding:3px 10px;font-size:13px}@media(max-width:800px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}h1{font-size:27px}}@media print{body{background:white}.box{break-inside:avoid}.controls{display:none}}
</style><main><div class="muted">CIRA · 可运行代码示例 · CPU</div><h1>看一次更新怎样发生</h1>
<div class="scope">这是修补后代码实际运行的玩具示例：小型网络＋合成数据，分数是 synthetic_quality_NOT_AP。四个逻辑客户端在同一CPU进程中顺序推进；图中连线来自演示交换记录，不代表四台Jetson实测。</div>
<div class="cards" id="cards"></div>
<div class="grid"><section class="box"><h2>客户端之间发生了什么</h2><svg id="network" viewBox="0 0 620 230"></svg><div class="muted">线上的数字为演示协议中记录的单向字节总数；没有发生记录的方向不画线。</div><details><summary>展开逐方向字节表</summary><div class="scroll" id="edgeTable"></div></details></section>
<section class="box"><h2>这个示例怎么设定</h2><div id="settings"></div><p class="muted">每个D批次有2个合成样本。这里每个epoch只有1个批次，epoch数字用于看清预算逻辑，不能当论文训练规模。</p><a href="demo_config.json">配置</a> · <a href="demo_data.json">精简记录</a> · <a href="verification.json">核对结果</a></section></div>
<section class="box"><h2>逐步查看一个客户端阶段</h2><div class="controls"><label>客户端 <select id="client"></select></label><label>阶段 <select id="stage"></select></label><span class="pill" id="release"></span><a id="raw">原始动作日志</a></div>
<p class="muted">默认显示操作种类最多的阶段，所有阶段均保留，可在上方切换。</p><div id="budget"></div><div class="bar"><div class="fill" id="fill"></div></div><div class="steps-label"><span><i class="dot" style="background:#537ea8"></i>Fresh / Refine</span><span><i class="dot" style="background:#d08a60"></i>协作成员修改</span><span><i class="dot" style="background:#8f7eaa"></i>完整状态Import</span></div>
<div class="flow" id="flow"></div><div class="detail" id="detail"></div>
<div class="scroll"><table><thead><tr><th>步</th><th>操作</th><th>触发规则</th><th>拟合批次 实际/准入</th><th>外部路径</th><th>V分数</th><th>相对父状态增益</th><th>阶段累计</th></tr></thead><tbody id="actions"></tbody></table></div></section>
<div class="grid"><section class="box"><h2>最后为什么发布，或为什么保留</h2><div id="audit"></div><p class="muted">V用于候选选择；锁定后才使用A审计。未通过则保留阶段入口状态。零增益和未发布都按实际结果显示。</p></section><section class="box"><h2>怎么再次运行</h2><pre id="command"></pre><a href="README_示例.md">完整使用说明</a><p class="muted">运行同一命令时请换一个新的输出目录，避免覆盖这次记录。浮点/软件环境和CPU耗时可能影响精确重放。</p></section></div>
<footer>每个动作可追溯到 run/client*/stage*/actions.jsonl；检查点、包、账本和独立报告也在run目录。这里的设备检查使用TinyBase的CPU分支，不用于估计Orin内存、延迟或功率。</footer>
<script id="payload" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.querySelector('#payload').textContent),$=s=>document.querySelector(s),E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=6)=>v===null||v===undefined?'缺失':String(Number(Number(v).toPrecision(n))),names={fresh:'Fresh 新配置',refine:'Refine 继续拟合',add:'Add 加入组件',remove:'Remove 移除组件',replace:'Replace 替换组件',import:'Import 完整状态',discrete:'离散接收'};
const color=k=>k==='import'?'#8f7eaa':['fresh','refine'].includes(k)?'#537ea8':'#d08a60';
$('#cards').innerHTML=[['逻辑客户端',D.clients],['环境阶段',D.stages],['实际动作',D.summary.attempts],['发布组件版本',D.summary.published_versions]].map(([t,n])=>`<div class="box"><div class="muted">${t}</div><div class="big">${n}</div></div>`).join('');
$('#settings').innerHTML=`每阶段累计上限 <b>${D.config.fitting.stage_epochs} epoch</b><br>本地 / 接收块最多 <b>${D.config.fitting.local_epochs} / ${D.config.fitting.receiving_epochs} epoch</b><br>最多 <b>${D.config.attempts}</b> 次动作，档案最多 <b>${D.config.archive_capacity}</b> 个状态<br>种子 <b>${D.config.seed}</b>，PCG64命名随机流`;
if(D.runs.every(r=>r.actions.every(a=>a.gain===0)))$('#settings').innerHTML+='<p class="muted">本次V增益均为0，适合观察状态与预算流程。阈值为0时，审计分数持平也可能发布新状态。</p>';$('#command').textContent=D.command;
const cs=[...new Set(D.runs.map(x=>x.client))];$('#client').innerHTML=cs.map(c=>`<option value="${c}">C${Number(c)+1}</option>`).join('');$('#stage').innerHTML=Array.from({length:D.stages},(_,i)=>`<option>${i+1}</option>`).join('');
function drawNet(){let n=D.clients,svg='';const pos=Array.from({length:n},(_,i)=>[60+(n===1?0:i*500/(n-1)),105]);svg+='<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6" fill="#8ba8b9"/></marker></defs>';for(const edge of D.edges){let a=pos[Number(edge.sender)],b=pos[Number(edge.receiver)];if(!a||!b)continue;let dir=Math.sign(b[0]-a[0]),up=Number(edge.sender)<Number(edge.receiver),offset=(up?-1:1)*(35+22*Math.abs(Number(edge.sender)-Number(edge.receiver)));svg+=`<path d="M${a[0]+dir*26},${a[1]} Q${(a[0]+b[0])/2},${105+offset} ${b[0]-dir*29},${b[1]}" fill="none" stroke="#9db8c6" stroke-width="2" marker-end="url(#arr)"/><text x="${(a[0]+b[0])/2}" y="${105+offset/2+(up?-5:17)}" font-size="11" text-anchor="middle" fill="#526f7f">${edge.bytes} B</text>`;}for(let i=0;i<n;i++)svg+=`<circle cx="${pos[i][0]}" cy="105" r="25" fill="#ecf3f7" stroke="#6086a4" stroke-width="2"/><text x="${pos[i][0]}" y="111" text-anchor="middle" font-size="17" fill="#33516a">C${i+1}</text>`;if(!D.edges.length)svg+='<text x="310" y="190" text-anchor="middle" fill="#657985">本次没有产生跨客户端交换</text>';$('#network').innerHTML=svg}drawNet();$('#edgeTable').innerHTML='<table><tr><th>发送方</th><th>接收方</th><th>单向字节</th></tr>'+D.edges.map(e=>`<tr><td>C${Number(e.sender)+1}</td><td>C${Number(e.receiver)+1}</td><td>${e.bytes} B</td></tr>`).join('')+'</table>';
let current;
function showStep(i){const a=current.actions[i];if(!a){$('#detail').textContent='本阶段未执行候选动作。';return}document.querySelectorAll('.step').forEach((n,j)=>n.classList.toggle('active',i===j));$('#detail').innerHTML=`<b>${E(names[a.kind]||a.kind)}</b> · ${E(a.status)}<br>父状态 <code>${E(a.parent)}</code><br>子状态 <code>${E(a.child)}</code><br>预测增益 ${fmt(a.prediction)}，覆盖项 ${fmt(a.coverage)}，优先级 ${fmt(a.priority)}<br>这一步实际尝试 ${a.fit_steps} 个拟合批次，完成 ${a.completed_fit_steps} 个；外部路径 ${a.packages.length} 条。${a.error?'<br>'+E(a.error):''}`;}
function update(){current=D.runs.find(r=>r.client==$('#client').value&&r.stage==$('#stage').value);const r=current,used=r.actions.reduce((s,a)=>s+a.fit_steps,0),cap=D.config.fitting.stage_epochs;$('#budget').textContent=`累计拟合 ${used} / ${cap} 批次；完成V评价 ${r.info.actual.search.E} 次；本地演示通信 ${r.info.actual.search.B} B；停止原因 ${r.info.stop}`;$('#fill').style.width=Math.min(100,used/cap*100)+'%';$('#release').textContent=r.info.audit.released?'发布了新部署状态':'保留入口状态';$('#raw').href=r.raw;$('#flow').innerHTML=r.actions.map((a,i)=>`<div class="step" data-i="${i}" style="border-color:${color(a.kind)}"><b>${i+1}. ${E(a.kind)}</b><small>拟合 ${a.fit_steps} / ${a.fidelity}</small><small>gain ${fmt(a.gain,4)}</small></div>`).join('');document.querySelectorAll('.step').forEach(x=>x.onclick=()=>showStep(Number(x.dataset.i)));$('#actions').innerHTML=r.actions.map(a=>`<tr><td>${a.attempt}</td><td>${E(a.kind)}</td><td>${E(a.decision)}</td><td>${a.fit_steps} / ${a.fidelity}</td><td>${a.packages.length}</td><td>${fmt(a.score)}</td><td>${fmt(a.gain)}</td><td>${a.stage_fit_steps}</td></tr>`).join('');const a=r.info.audit;$('#audit').innerHTML=`A中候选分数：<b>${fmt(a.candidate_score)}</b><br>A中入口分数：<b>${fmt(a.incumbent_score)}</b><br>差值：<b>${fmt(a.difference)}</b>，阈值：${fmt(a.threshold)}<br>演示设备检查：${a.device_pass?'通过':'未通过'}<br>最终部署：<code>${E(r.info.deployed.slice(0,18))}…</code><br>最终部署的外部路径：${r.info.deployed_external_paths??'缺失'}${a.error?'<br>'+E(a.error):''}`;showStep(0)}
$('#client').onchange=update;$('#stage').onchange=update;const first=[...D.runs].sort((a,b)=>new Set(b.actions.map(x=>x.kind)).size-new Set(a.actions.map(x=>x.kind)).size)[0];$('#client').value=first.client;$('#stage').value=first.stage;update();
</script></main></html>'''


def run(output, clients=4, stages=2, seed=1001):
    import torch
    from softcas.cli import demo
    from softcas.config import load_config, validate_config
    if not 1 <= clients <= 8 or not 1 <= stages <= 4:
        raise ValueError("Small example supports 1–8 logical clients and 1–4 stages")
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new or empty example directory")
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / 'configs/fixture.json')
    config.update(seed=seed, proposal_protocol='native_r1000', attempts=12, archive_capacity=8,
                  fitting=dict(unit='epochs', stage_epochs=12, local_epochs=4, receiving_epochs=2))
    validate_config(config)
    torch.set_num_threads(1)
    (output/'demo_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    summary = demo(config,output/'run',stages=stages,clients=clients)
    assert summary['purpose']=='SYNTHETIC_BEHAVIOR_ONLY' and summary['metric']=='synthetic_quality_NOT_AP'
    runs=[];checks=0
    for p in sorted((output/'run').glob('client*/stage*/stage.json')):
        info=json.loads(p.read_bytes())
        actions=[json.loads(s) for s in p.with_name('actions.jsonl').read_bytes().splitlines()]
        used=0
        for action in actions:
            used+=action['fit_steps']
            assert used==action['stage_fit_steps']<=action['stage_fit_step_limit']==12
            assert action['fit_steps']<=action['fidelity']
            if action['status']=='failed':assert action['gain']==0 and action['score'] is None
            checks+=1
        ledger=[json.loads(s) for s in p.with_name('ledger.jsonl').read_bytes().splitlines()]
        import math
        for scope in ('search','release'):
            for key in ('F','E','B','W'):
                assert math.isclose(sum(e['cost'][key] for e in ledger if e['scope']==scope),
                                    info['actual'][scope][key],rel_tol=1e-12,abs_tol=1e-9)
        assert info['actual']['search']['E']<=1+len(actions)
        assert info['deployed']==info['audit']['deployed']
        from softcas.state import State
        info['deployed_external_paths']=len(State.load(p.parent/'checkpoints'/(info['deployed']+'.json')).C)
        runs.append(dict(client=p.parent.parent.name[6:],stage=int(p.parent.name[5:]),
                         raw=p.with_name('actions.jsonl').relative_to(output).as_posix(),info=info,actions=actions))
    network=json.loads((output/'run/network_transfers.json').read_bytes());edges={}
    for record in network.values():
        key=(record['sender'],record['receiver']);edges[key]=edges.get(key,0)+record['bytes']
    assert sum(edges.values())==summary['network_tx_bytes']
    command=f'python examples/run_cira_demo.py --output examples_output/new_demo --clients {clients} --stages {stages} --seed {seed}'
    data=dict(clients=clients,stages=stages,summary=summary,config=config,runs=runs,command=command,
              edges=[dict(sender=a,receiver=b,bytes=n) for (a,b),n in edges.items()])
    text=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    (output/'demo_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'打开示例.html').write_text(PAGE.replace('__DATA__',text),encoding='utf-8')
    with (output/'逐步动作.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['client','stage','attempt','kind','decision','fidelity','fit_steps','completed_fit_steps','score','gain','stage_fit_steps','parent','child']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in runs:
            for a in r['actions']:writer.writerow({k:r[k] if k in ('client','stage') else a[k] for k in fields})
    verification=dict(scope='CPU_SYNTHETIC_BEHAVIOR_ONLY_NOT_AP',clients=clients,stages=stages,
                      client_stages=len(runs),actions_checked=checks,epoch_caps_checked=True,
                      ledger_reconciled=True,deployment_identity_checked=True,network_bytes_checked=True,
                      implementation_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'softcas').glob('*.py')})
    (output/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    readme=f'''# CIRA可运行小示例

先打开 `打开示例.html`，切换客户端和阶段，再点击动作卡片。界面来自本次实际执行日志，不预设动作种类或收益。

当前运行：{clients}个逻辑客户端、{stages}阶段、{len(runs)}个客户端阶段、{checks}次动作。全部在同一CPU进程执行，数据和TinyBase均为玩具示例，分数不是AP，不代表YOLO或四Jetson性能。

每阶段最多12次动作、累计12个D epoch，本地块4/接收块2 epoch。这个D每epoch只有一个批次（2个合成样本）。Fresh/Refine、协作与Import是否发生，取决于实际候选、预算与选择规则。

## 再次运行

在softcas_impl目录，用已有CPU环境：

```powershell
.\\.venv\\Scripts\\python.exe examples/run_cira_demo.py --output examples_output/new_demo --clients {clients} --stages {stages} --seed {seed}
```

已生成的非空目录会被拒绝覆盖。输出配置是SYNTHETIC_BEHAVIOR_ONLY；没有修改native_r1000正式参考模板。

## 读图与文件

- Fresh：从父状态构建新配置，重新初始化本地适配器。
- Refine：保留配置，继续拟合。
- Add/Replace/Remove：修改外部成员；最后成员Remove可为零拟合。
- Import：恢复完整已有状态，零拟合后在当前上下文评价。
- gain：child V减parent V；不是相对阶段最佳的提升，也不是AP。
- 最终发布：先锁定V选中的状态，再比较A和演示设备检查；失败或不满足条件时保留入口。

`逐步动作.csv`供表格阅读；`run/client*/stage*/actions.jsonl`含完整特征与种子；相邻的ledger/checkpoints/packages保存成本与状态。`run/test_matrix.jsonl`是独立报告，未反馈给控制器。`verification.json`记录预算、账本、部署和网络核对。CPU时间与部分算子F计数不是设备标定。
'''
    if all(a['gain']==0 for r in runs for a in r['actions']):
        readme+='\n本次V增益均为0，主要用于观察状态与预算。A阈值为0时，持平也允许发布新状态。\n'
    (output/'README_示例.md').write_text(readme,encoding='utf-8')
    print(json.dumps(dict(entry=str(output/'打开示例.html'),client_stages=len(runs),actions=checks,
                         published_versions=summary['published_versions'],scope=verification['scope']),ensure_ascii=False))
    return verification


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--clients',type=int,default=4)
    p.add_argument('--stages',type=int,default=2)
    p.add_argument('--seed',type=int,default=1001)
    a=p.parse_args();run(a.output,a.clients,a.stages,a.seed)
