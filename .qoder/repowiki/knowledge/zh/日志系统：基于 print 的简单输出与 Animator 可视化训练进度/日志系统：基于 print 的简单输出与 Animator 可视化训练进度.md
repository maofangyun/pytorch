---
kind: logging_system
name: 日志系统：基于 print 的简单输出与 Animator 可视化训练进度
category: logging_system
scope:
    - '**'
source_files:
    - chapter_linear-networks/softmax-regression-scratch.ipynb
    - chapter_multilayer-perceptrons/mlp-concise.py
    - chapter_multilayer-perceptrons/mlp-scratch.py
    - chapter_recurrent-modern/lstm.py
    - chapter_recurrent-modern/seq2seq.py
    - chapter_convolutional-neural-networks/lenet.ipynb
    - chapter_attention-mechanisms/nadaraya-waston.ipynb
---

本仓库是《动手学深度学习》PyTorch 版的教程集合，**并未引入任何专门的日志框架（如 logging、loguru、structlog 等）**。整个项目的“日志/输出”完全依赖 Python 内置的 `print()` 以及 d2l 库提供的 `Animator` 类进行训练过程的可视化展示。

### 1. 使用的系统与工具
- **标准输出（stdout）**: 所有训练循环中的 epoch、loss、accuracy、速度等信息均通过 `print(f'...')` 直接打印到控制台。
- **d2l.Animator**: 用于在 Jupyter Notebook 中增量绘制训练曲线（损失、准确率等），替代传统日志文件记录数值指标。
- **assert 断言**: 部分训练函数末尾使用 `assert` 对训练结果做基本合理性检查，作为轻量级的“运行时日志校验”。

### 2. 关键文件与位置
- `chapter_linear-networks/softmax-regression-scratch.ipynb`: 定义 `train_epoch_ch3`、`train_ch3`、`Animator` 等核心训练与可视化函数，被后续章节广泛复用。
- `chapter_multilayer-perceptrons/mlp-concise.py`、`mlp-scratch.py`: 使用 `print(f'epoch {epoch + 1}, loss ..., train acc ..., test acc ...')` 格式输出训练进度。
- `chapter_recurrent-modern/lstm.py`、`seq2seq.py`: 同样以 `print(f"Epoch [{epoch+1}/{epochs}], Loss: ...")` 形式输出。
- `chapter_convolutional-neural-networks/lenet.ipynb`、`chapter_attention-mechanisms/nadaraya-waston.ipynb` 等大量 Notebook 中，训练循环内嵌 `print` 语句。

### 3. 架构与约定
- **无集中式日志配置**: 没有 `logging.basicConfig`、logger 实例化或配置文件，每个 Notebook/脚本自行决定输出内容。
- **结构化程度低**: 输出为人类可读的 f-string 文本，非 JSON/结构化格式，无法被外部日志系统解析。
- **训练指标可视化优先**: 对于需要长期跟踪的指标（loss、acc），优先使用 `Animator.add(epoch, value)` 在 Notebook 中实时绘图，而非写入日志文件。
- **错误诊断依赖 assert**: 训练函数末尾常用 `assert train_loss < 0.5` 等断言快速验证训练是否“合理收敛”，失败时抛出异常并中断执行。

### 4. 开发者应遵循的规则
- **不要引入额外日志框架**: 本项目定位为教学示例，保持代码简洁，避免分散注意力。
- **统一使用 print 风格**: 如需新增输出，建议沿用 `f'epoch {epoch + 1}, loss {float(l.sum()):.6f}'` 这类简洁格式。
- **训练指标优先用 Animator**: 可复用的训练函数应返回 metric 并通过 `animator.add` 可视化，而不是仅靠 print 查看。
- **谨慎使用 assert**: 仅在调试阶段保留，正式发布或批量实验时应移除或改为条件日志。
- **避免在生产环境直接使用此模式**: 若需部署，应替换为标准 `logging` 模块或结构化日志方案。

### 总结
该仓库**不存在正式的 logging_system**，日志输出完全由分散的 `print()` 和 `Animator` 可视化构成，属于教学型代码的典型风格——简单、直观、无需配置。