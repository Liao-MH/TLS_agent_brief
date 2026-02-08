# TLS_agent_brief: 运行说明（更新版）

## 1) 你需要做的第一件事：确认路径策略
- 当前脚本与配置**默认保留原始 HPC 路径**（/lustre1、/home/...）。
- 若要本机运行，请通过环境变量覆盖：
  - `PROJECT_ROOT`（TLS_agent_brief 根）
  - `DATA_ROOT`（数据）
  - `OUTPUT_ROOT`（输出）
  - `MMSEG_ROOT`（mmseg 源码/安装位置）
  - `CHECKPOINT_ROOT`（权重）
  - `WSI_PATH`（WSI 输入）

## 2) 缺失/外部依赖在哪里？
已汇总为两个文件（含文件+行号定位）：
- `MISSING_CONSOLIDATED.json`
- `MISSING_READABLE.md`
如需原始抽取清单，可参考 `MANIFEST.json`

## 3) 环境重建
- 如果你使用 pip：`pip install -r requirements.txt`
- 如果你使用 conda：建议同时导出 `conda_env.yml`（见主 spec）

## 4) 迭代补齐策略
运行入口脚本 -> 根据报错补齐：config / python module / 权重 / 系统库。
或直接按 `MISSING_READABLE.md` 的位置清单逐项补齐。
