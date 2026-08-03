---
kind: configuration_system
name: 配置系统：无专用配置框架，依赖 Notebook 内联参数与环境变量
category: configuration_system
scope:
    - '**'
source_files:
    - setup.py
    - TERMINOLOGY.ipynb
---

该仓库是《动手学深度学习》PyTorch 版的教程集合，**不存在统一的运行时配置系统**。代码以 Jupyter Notebook 为主，所有超参数、路径、模型设置等均以硬编码或 Notebook 单元格变量的形式直接写在每个章节的 notebook 中，没有集中式的配置文件（如 .yaml/.toml/.env）或配置加载模块。

具体表现：
- **setup.py** 仅声明包元数据与依赖（jupyter、numpy、matplotlib、requests、pandas），不包含任何配置解析逻辑。
- 各章节 Python 脚本（如 `chapter_multilayer-perceptrons/mlp-concise.py`、`chapter_deep-learning-computation/parameters.py`）通过 `from d2l import torch as d2l` 导入 d2l 库，但 d2l 本身在本书中作为教学辅助库提供绘图、数据加载、训练循环等工具，并非配置中心。
- 搜索 `.env`、`os.environ`、`getenv`、`argparse`、`configparser`、`pyyaml`、`toml` 等常见配置模式，除 `uv.lock` 中因 jupyter 生态间接依赖 pyyaml/tomli 外，**没有任何代码实际使用这些库进行配置读取**。
- 附录中的 `aws.ipynb`、`jupyter.ipynb` 涉及 AWS EC2/SageMaker/Jupyter 的环境配置，但这些是外部平台的使用说明，不属于应用自身的配置系统。

因此，本仓库**不具备传统意义上的配置系统**，开发者如需修改实验参数，需直接编辑对应 Notebook 或脚本中的变量。