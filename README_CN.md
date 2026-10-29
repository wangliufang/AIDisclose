# AIDisclose 中文使用说明

AIDisclose 是一个 Python 软件包，用于从企业报告中检索人工智能相关表述，
识别披露语句，并生成企业年度及分组指标。软件源码、使用文档、示例、词典、
测试数据和测试程序均包含在此目录中。

## 环境与安装

需要 Python 3.10 或更高版本。解压后打开终端，进入本文件所在的 `AIDisclose`
目录，执行：

```bash
python -m pip install .
aidisclose doctor
```

安装监督学习及 PDF 支持：

```bash
python -m pip install ".[ml,pdf]"
```

安装时，pip 需要取得 `pyproject.toml` 中列出的依赖；若本机没有这些依赖，
需要可访问相应的 Python 软件包源。请根据系统环境使用 `python3` 或 `py`
替代上述 `python` 命令。

## 运行演示

```bash
aidisclose demo
```

演示使用软件自带的虚构企业报告，适合离线了解完整流程；演示结果不能作为
真实企业的实证研究结论。更多示例请参见 `examples/`。

## 主要文件

- `src/aidisclose/`：软件源码、词典和演示数据。
- `docs/`、`annotation/`：使用手册、方法说明和标注规则。
- `examples/`：典型操作示例。
- `tests/`：功能测试及测试数据。
- `dist/`：基于本包源码构建的 Python 安装包。
- `LICENSE`、`CITATION.cff`：许可和引用信息。

恢复已有运行时，词典文件和检查点需要与原始记录中的 SHA-256 相匹配；
无法匹配时请保留原始词典，或使用新的检查点目录重新运行。

使用 XBRL 数据构建年度面板前，应人工确认报告的实际覆盖期间；软件不能自动
保证每份报告均为完整年度报告。模型标签反映披露中的表述，不等于企业真实
采用情况。研究分析应使用独立标注数据验证，并结合 `docs/validation.md`
理解相关统计方法的适用范围和局限。

## 作者

* **Liufang Wang** —— 岭南大学社会科学学院，韩国庆尚北道庆山市 38541。邮箱：qq1055145729@gmail.com
* **Xiangyang Sun**\* —— 昌信大学工商管理学院，韩国庆尚南道昌原市马山会原区巴龙路 262，51352。邮箱：sunx200002@gmail.com

\* 通讯作者。
