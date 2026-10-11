# GT-IDEA: toughness and isolated toughness variants

本文实验代码和数据的独立上传包，对应论文 *A Group-Theoretic I-Ching Evolutionary Algorithm for Computing Toughness and Isolated Toughness Variants*。

本文件夹可整体上传到 GitHub 仓库根目录。这里包含原始实验记录和本次审稿补充实验记录；整理过程中未重新运行论文实验、未修改结果数值。原工作目录中的文件未移动或删除。

## 目录

| 目录/文件 | 内容 |
| --- | --- |
| `code/` | 核心算法、基线、历史实验及补充实验驱动脚本 |
| `data/archived/` | 原始实验的 21 个 CSV/统计文本文件，保留原文件名 |
| `data/supplementary/revision_20261011/` | 110 次补充实验的原始记录、收敛历史、汇总、环境信息和既有核验报告 |
| `figures/` | 论文补充收敛图的 PDF/PNG 原文件 |
| `checks/` | 文件校验和可独立运行的科学检查 |
| `docs/EXPERIMENT_MAP.md` | 论文表格/图与脚本、数据的对应表 |
| `docs/DATA_PROVENANCE.md` | 数据来源、版本区别、种子与计数限制 |
| `requirements.txt` | 本地已安装并用于检查的依赖版本 |
| `manifest.json` | 文件大小、SHA-256、来源及是否与原文件完全一致 |

算法版本：`GTIDEAv2` 是当前论文主要算法，增加局部下降；它继承 `gt_idea.py` 的基础实现。原始 `GTIDEA` 用于原版实验、消融及探索性实验，两个核心文件都必须保留。

## 安装和快速检查

在本文件夹根目录运行（本地检查环境为 Python 3.12.7）：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python checks/verify_bundle.py
python checks/scientific_checks.py
```

科学检查只产生 `outputs/verification/` 中的检查报告，不修改归档数据，也不是论文完整实验的重跑。

## 运行当前论文相关实验

```bash
# 小图精确值与 v2 解的比较，10 次重复
python code/exp1b_v2_accuracy.py --runs 10 --seed 0 --out outputs/accuracy

# 主比较：v2 与 GA / SA / BPSO，30 次重复
python code/exp9_multibaseline.py --runs 30 --seed 0 --out outputs/main

# v2 / 原版消融 / GA，30 次重复
python code/exp_v2_comparison.py --runs 30 --seed 0 --out outputs/ablation

# 大型对称图，30 次重复
python code/exp8_v2_large_symmetric.py --runs 30 --seed 0 --out outputs/symmetric

# 原版可扩展性和参数敏感性：CLI 内部以 runs//3 执行 Exp5，故 30 对应 10 次
python code/experiments.py --exp 5 --runs 30 --seed 0 --out outputs/exploratory

# 审稿补充实验：80 次收敛实验 + 30 次额外阈值实验，共 110 次
python code/revision_experiments.py --runs 10 --generations 30 --population 20
```

完整实验耗时较长。新的运行结果进入 `outputs/`，与论文已有数据分开；补充脚本也支持 `--out` 和 `--figures` 自定义目录。请不要把输出路径指向 `data/`。历史的 Exp1--6 和演示脚本同样保留，详情见对应表。

## 结果的解释

当前论文已说明历史 v2 比较的实际适应度调用次数不相等：`experiments.run_gtidea_counted` 总是实例化原版 `GTIDEA`，基线以该参考运行标定预算，而 v2 单独按固定代数运行局部下降。归档结果可用于报告所实现分配下的解质量，不能据此声称等 FE 下效率更高。整理包保留这套历史运行逻辑，没有悄悄改成新实验。

Exp9 原始 CSV 只有目标值，没有 FE/时间列；缺失测量未补造。原始可扩展性/参数实验仅保存汇总，缺失的逐次运行记录同样未补造。补充实验记录每次目标调用（含缓存命中）、唯一目标计算、预处理时间、搜索时间和逐代最优值。

合成图和小图由代码中的 NetworkX 生成器构造，真实网络使用 NetworkX 内置数据集，无需另外下载。本地没有独立的大型不规则真实网络文件；论文该部分采用性能分析，没有新增对应实验数据。

## 上传

只上传本文件夹内的内容即可。论文 LaTeX、作者照片、Response、模板、备份、排版脚本和缓存不属于此实验包。`outputs/` 已加入 `.gitignore`，避免本地试跑结果混入归档。

文件来源和所有上传文件的 SHA-256 见 `manifest.json`。数据/图文件均逐字节复制，算法核心未改动；为便于独立运行，少量驱动脚本只调整了路径、命令行入口或历史说明。具体变化见 `docs/DATA_PROVENANCE.md`。
