# 上传清单与操作指引

## 你要传的就是这个文件夹

```
C:\Users\Lenovo\WorkBuddy\2026-10-07-22-28-57\github_repo\
```

**39 个文件，568 KB** —— 远低于 GitHub 网页上传 100 个文件 / 单文件 100 MB 的限制。

已清理干净：无 `__pycache__`、无 `venv`、无 `.env`、无密钥文件。

## 上传步骤

1. 打开（收藏夹里应该有）：

   ```
   https://github.com/frostvog/sodium-aidc-wireless-power/upload
   ```

2. 把整个 `github_repo` 文件夹**拖到页面上传区**

3. 页面下方 commit 说明填：

   ```
   P0-P12 全流程：非相干合成无线供电从 20W 外推至机柜级 3-30kW

   - 19 个源模块：L1 钠电缓冲 / L2 无线场域 / C1 仲裁
   - 25 个测试，22 通过 3 失败（失败项故意保留，见 README「已知问题」）
   - 55 处TODO[实测] 标记待标定项
   - 引用 Aalto 2024 (CC BY 4.0) 实测参数与 4x4 相位图案
   ```

4. 点 **Commit changes**

5. 复制地址栏链接，贴到社区

## 文件清单

| 类别 | 数量 | 内容 |
|---|---|---|
| `src/` | 19 个 `.py` | 全部业务代码 |
| `tests/` | 9 个 `.py` | 单元测试 |
| `scripts/` | 2 个 `.py` | P7寻优 / P8 削峰报告生成器 |
| `outputs/` | 4 个 | 盲区分布图、Pareto 摘要、削峰对比 |
| 根目录 | 4 个 | README、LICENSE、requirements.txt、.gitignore |

## 我做过的事（改了什么）

| 项 | 说明 |
|---|---|
| **修了一个真实 bug** | `src/l1_buffer/peak_shave.py` 第 124 行把变量 `threshold_kw` 当函数调用，导致 `run_p8_report.py` 直接崩溃。已修复并实跑验证。 |
| 剔除缓存与依赖 | 删掉 `__pycache__`，未包含 venv |
| 补齐README | 按社区读者视角写：开头一句话说清是什么、为什么值得看 |
| 补齐配置文件 | `.gitignore`、`requirements.txt`、`LICENSE`（MIT + CC BY 归属说明） |
| 保留 P7 大CSV 未传 | `p7_pareto_front.csv` 有 2 MB，可由 `scripts/run_p7_report.py` 重新生成，不必入库 |

## 三个失败测试的处置

**不要去修它们，也不要改断言数值。**

Codex 在 `tests/test_solver.py` 里留了注释，说明这些失败是故意保留的，
目的是暴露方法局限而非掩盖。把数字改小会让整份README 的诚实性失效。

三个失败的性质（详见 README）：

| 测试 | 实测 | 要求 | 性质 |
|---|---|---|---|
| `test_load_independence` | 2.10e-04 | < 1e-06 | 判据本身不可达（差2 个数量级） |
| `test_no_blind_spot` | 6.22e-18 | ≥ 0.5 | 互感 kernel 在 y=2.0 对称退化 |
| `test_non_four_cluster_boundary` | 6.80e-17 | ≥ 0.5 | 同上 |

## 复现验证

```bash
cd github_repo
pip install -r requirements.txt
python -m unittest discover -s tests     # 预期：Ran 25, FAILED (failures=3)
python scripts/run_p8_report.py         # 预期：输出三档削峰对比 JSON
```

两个脚本均已实跑确认可用。
