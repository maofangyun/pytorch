---
kind: build_system
name: 构建与依赖管理：基于 setuptools 的 Python 包发布与 uv 锁定
category: build_system
scope:
    - '**'
source_files:
    - setup.py
    - uv.lock
    - .gitignore
---

该仓库是一个以 Jupyter Notebook 为主的深度学习教程项目，其构建系统相对简单，主要围绕 Python 包打包和依赖管理展开。

**核心构建工具**
- `setup.py`：使用 setuptools 定义 d2l 包的元数据（名称、版本 1.0.3、Python>=3.5）、依赖项（jupyter、numpy、matplotlib、requests、pandas）和包发现规则
- `uv.lock`：使用 uv 包管理器生成的锁文件，锁定所有依赖的精确版本，支持多 Python 版本（3.9-3.14）的条件解析
- `.gitignore`：忽略构建产物和临时文件

**依赖管理策略**
- 运行时依赖通过 setup.py 的 install_requires 声明
- 开发环境依赖通过 uv.lock 精确锁定，确保可重复性
- 支持多平台 wheel 分发（manylinux、macOS、Windows）
- 条件依赖根据 Python 版本自动选择合适版本

**构建流程特点**
- 无 Makefile、Dockerfile 或 CI/CD 配置
- 无测试框架集成（pytest/unittest）
- 无代码质量检查工具链
- 纯 Python 包发布模式，通过 `python setup.py sdist bdist_wheel` 构建

**开发者注意事项**
- 新增依赖需同时更新 setup.py 和运行 `uv lock` 保持锁文件同步
- 包版本遵循语义化版本控制（当前 1.0.3）
- 由于是教程项目，构建系统保持极简，重点在于内容而非工程化