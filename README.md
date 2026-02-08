# TLS_agent_brief

最小可运行子集（自动抽取）说明。

## 结构
- `scripts/`：已完整复制自原仓库（入口脚本在此）
- `TLS_agent/`：根据脚本/配置静态分析复制到的最小代码与配置
- `TLS-Configs/`、`tools/`、`mmseg/`、`projects/`、`tls_toolkit/`、`configs/`：入口脚本与配置所需代码
- `MANIFEST.json`：缺失文件与外部依赖清单
- `MISSING_CONSOLIDATED.json`：缺失项汇总（含文件与行号定位）
- `MISSING_READABLE.md`：缺失项的可读版本（含文件与行号定位）
- `requirements.txt`：从原仓库 `TLS_agent/requirements.txt` 复制

## 运行前必须处理
- 修正脚本中的硬编码路径（HPC/旧目录），统一使用：
  - `PROJECT_ROOT`（本目录）
  - `DATA_ROOT`（数据）
  - `OUTPUT_ROOT`（输出）
  - `MMSEG_ROOT`（mmseg 源码或安装位置）
  - 当前已保持**原始 HPC 路径为默认值**，如需本机运行请通过环境变量覆盖

## 缺失/外部依赖
优先查看（已包含定位信息）：
- `MISSING_CONSOLIDATED.json`：完整缺失项 + 位置
- `MISSING_READABLE.md`：人类可读摘要 + 位置
如需原始抽取清单，可参考 `MANIFEST.json`

## 环境
- `requirements.txt` 来自原仓库，仅作为基础依赖参考。
- 若需可复现环境，建议在你的运行环境中重新导出：
  - `pip freeze > requirements.txt`
  - `conda env export --no-builds > conda_env.yml`

## 入口脚本
- `scripts/4_Train_Mask2Former_Morph.sh`
- `scripts/5-4_Train_best_Morph_config.sh`
- `scripts/7_TLS_agent_toolkit.sh`
