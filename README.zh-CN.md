# 基于 XGBoost 的共享单车出行绿地暴露可解释分析

[English](README.md)

## 项目概述

本仓库提供共享单车出行绿地暴露的模型训练、精度评价及 SHAP/PDP 解释代码。
六种模型各有一个独立文件，SHAP 和 PDP 合并在一个文件中。
每个文件均可直接运行，路径、目标和参数集中在文件开头。

## 研究目标

- 预测绿地暴露及其空间分布；
- 分析时间、空气污染和气象变量对模型预测的贡献；
- 通过 SHAP 和 PDP 展示非线性关系与变量组合下的模型响应。

## 数据来源

出行数据来自 [Mobike/Biendata](https://www.biendata.xyz/)，绿地信息来自
[ESA WorldCover](https://esa-worldcover.org/en)，空气污染观测来自
[中国环境监测总站](https://www.cnemc.cn/)，气象数据来自
[中国气象数据服务中心](https://data.cma.cn/)。

将整理匹配的小时表存放在本地 `data/GE.csv`，格式见 [数据说明](data/README.md)。

## 输入变量

预测变量：星期、小时、O3、PM2.5、PM10、SO2、NO2、CO、温度、风速、气压、湿度、降水。
对应的英文列名和顺序在每个脚本的 `FEATURES` 中列出。

四个目标分别训练：`SGE`、`EGE`、`GiniStart`、`GiniEnd`，对应起点与终点绿地暴露及其基尼系数。

## 仓库结构

| 文件 | 用途 |
| --- | --- |
| `OLS.py` | 普通最小二乘回归 |
| `RF.py` | 随机森林 |
| `CatBoost.py` | CatBoost |
| `LightGBM.py` | LightGBM |
| `XGBoost.py` | XGBoost |
| `LSTM.py` | LSTM |
| `SHAP_PDP.py` | SHAP 总结图、依赖图、一维与二维 PDP |

七个代码文件均不依赖仓库中的其他 Python 文件。
模型结果保存在 `results/模型名/`，解释结果保存在 `results/SHAP_PDP/模型名/`。

## 软件环境

建议 Python 3.12，在仓库目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 使用 `.venv\Scripts\activate` 激活环境。

## 研究方法

### 1. 模型训练

按时间排序，将最后20%的原始记录作为测试时段，在训练期内进行五折扩展窗口验证。
候选配置按平均验证 MSE 选择，确定配置后拟合训练目标，再评价共同测试目标。

六模型使用相同的有效目标时刻，每个目标需要连续24小时输入。
LSTM 使用当前及此前23小时输入估计当前目标，属于后续时段的同小时回归。
缩放器仅拟合训练信息，最终测试时段不用于早停。OLS 和 LSTM 各使用一个配置。

### 2. 模型评价

每个模型导出训练与测试的 MAE、RMSE、MSE、R²、逐行预测和交叉验证结果，
并生成包含四个目标的实际值—预测值图，采用共享坐标名称和一个图例。

### 3. SHAP 分析

`SHAP_PDP.py` 读取已保存的表格模型和训练背景，计算全部变量的 SHAP 贡献与重要性，
生成总结图，并校验 SHAP 贡献之和与模型预测一致。

### 4. 一维依赖分析

同一文件生成 SHAP 依赖散点图和真正的一维 PDP。默认显示重要性前6个变量，
修改 `TOP_N = 13` 可分页展示全部变量。

### 5. 二维部分依赖分析

按训练 SHAP 排序，前三个变量分别与紧随其后的三个变量配对，形成3×3共9个子图。
每个网格点在全部训练背景行上平均模型预测，子图共用真实预测范围确定的色标。
图片导出600 dpi PNG、可编辑文字 PDF 和 SVG。

SHAP 和 PDP 描述模型行为，本身不构成因果效应或交互显著性的证据。

## 运行代码

选择所需模型直接运行：

```bash
python OLS.py
python RF.py
python CatBoost.py
python LightGBM.py
python XGBoost.py
python LSTM.py
```

修改每个文件开头的 `DATA_FILE`、`TARGETS`、`OUTPUT_DIR` 和模型参数即可调整运行内容。

先运行 `XGBoost.py` 保存模型，再运行合并的解释文件：

```bash
python SHAP_PDP.py
```

解释过程不会重新训练模型。修改 `MODEL_NAME` 可选择 OLS、RF、CatBoost、LightGBM 或 XGBoost。
`TARGETS`、`TOP_N`、`GRID_POINTS` 分别控制目标、展示变量数量和 PDP 网格数。

## 可重复性

各模型文件采用相同的变量顺序和时间划分规则，结果保存模型、指标、逐行预测和交叉验证分数。
表格模型文件同时保存用于解释的训练背景。

仅检查输入和有效样本数、不训练模型：

```bash
python XGBoost.py --check-only
```

GitHub Actions 检查七个文件的 Python 语法。

## 引用

在研究中使用本仓库时，请引用相关论文。

## 使用说明

本仓库用于学术研究和教育。使用代码时请引用相关论文。
