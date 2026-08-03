---
kind: dependency_management
name: Python 依赖管理：uv + setup.py 双轨制
category: dependency_management
scope:
    - '**'
source_files:
    - setup.py
    - uv.lock
    - pytorch.egg-info/requires.txt
---

本仓库采用 **两套并行的 Python 依赖管理机制**，分别服务于不同的使用场景：

## 1. 包发布与安装：setup.py（setuptools）
- `setup.py` 中通过 `install_requires` 声明运行时依赖：`jupyter, numpy, matplotlib, requests, pandas`
- 包名为 `d2l`，版本 `1.0.3`，要求 Python >= 3.5
- 构建产物位于 `pytorch.egg-info/`，其中 `requires.txt` 会生成可被 pip 识别的依赖清单
- 该方式主要用于将 d2l 作为可安装包发布到 PyPI 或本地安装

## 2. 开发环境锁定：uv.lock（uv 包管理器）
- 使用现代 Rust 编写的 `uv` 工具进行依赖解析与环境管理
- `uv.lock` 是完整的锁文件，包含所有直接和间接依赖的精确版本、哈希校验和多平台 wheel 信息
- 支持多 Python 版本条件解析（`resolution-markers`），覆盖 Python 3.9 到 3.14
- 默认要求 `requires-python = ">=3.12"`，比 setup.py 更严格
- 所有包源指向 `https://pypi.org/simple`，无私有仓库配置

## 关键约定与约束
- **依赖声明位置**：运行时依赖在 `setup.py` 中维护，开发/完整依赖由 uv 自动解析并锁定
- **版本策略**：uv.lock 提供确定性构建，确保不同环境下的依赖一致性
- **平台兼容性**：uv.lock 预编译了多平台 wheel（macOS x86_64/arm64、Linux manylinux、Windows），避免本地编译
- **无 vendoring**：不将第三方库源码纳入仓库，仅通过包管理器拉取
- **无 requirements.txt**：未使用传统的 pip 依赖文件，统一由 uv.lock 管理

## 开发者注意事项
- 新增依赖时需同时更新 `setup.py` 中的 `install_requires` 列表
- 运行 `uv lock` 重新生成 `uv.lock` 以锁定新依赖的确切版本
- 由于 uv.lock 已包含完整依赖树，无需额外维护 `requirements.txt` 或 `Pipfile`