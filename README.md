# TLS_agent_brief

最小可运行子集（自动抽取）说明。

## 结构
- `scripts/`：已完整复制自原仓库（入口脚本在此）
- `TLS_agent/`：根据脚本/配置静态分析复制到的最小代码与配置
- `MANIFEST.json`：缺失文件与外部依赖清单
- `requirements.txt`：从原仓库 `TLS_agent/requirements.txt` 复制

## 运行前必须处理
- 修正脚本中的硬编码路径（HPC/旧目录），统一使用：
  - `PROJECT_ROOT`（本目录）
  - `DATA_ROOT`（数据）
  - `OUTPUT_ROOT`（输出）
  - `MMSEG_ROOT`（mmseg 源码或安装位置）

## 缺失/外部依赖
查看 `MANIFEST.json`：
- `missing_files`：脚本/配置引用到但仓库内不存在
- `external_paths`：仓库外路径（默认未复制）

## 环境
- `requirements.txt` 来自原仓库，仅作为基础依赖参考。
- 若需可复现环境，建议在你的运行环境中重新导出：
  - `pip freeze > requirements.txt`
  - `conda env export --no-builds > conda_env.yml`

## 入口脚本
- `scripts/4_Train_Mask2Former_Morph.sh`
- `scripts/5-4_Train_best_Morph_config.sh`
- `scripts/7_TLS_agent_toolkit.sh`
