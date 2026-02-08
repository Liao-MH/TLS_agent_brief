# TLS_agent_brief: 运行说明（自动生成骨架）

## 1) 你需要做的第一件事：修脚本里的路径
- 优先把绝对 HPC 路径替换成：
  - PROJECT_ROOT（TLS_agent_brief 根）
  - DATA_ROOT（数据）
  - OUTPUT_ROOT（输出）
  - MMSEG_ROOT（mmseg 源码/安装位置）

## 2) 缺失/外部依赖在哪里？
请打开 `MANIFEST.json`：
- `missing_files`：脚本/配置引用到了，但 repo 内找不到（可能是旧路径或仓库外文件）
- `external_paths`：在 repo 外存在（例如你本机/别处盘符），默认不会复制进 brief

## 3) 环境重建
- 如果你使用 pip：`pip install -r requirements.txt`
- 如果你使用 conda：建议同时导出 `conda_env.yml`（见主 spec）

## 4) 迭代补齐策略
运行入口脚本 -> 根据报错补齐：config / python module / 权重 / 系统库。
