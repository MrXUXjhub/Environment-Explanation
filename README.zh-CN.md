# 基于 XGBoost 的共享单车出行绿地暴露可解释分析

[English](README.md)

## 项目概述

本仓库提供使用 **XGBoost** 和 **可解释人工智能（XAI）** 方法建模与解释共享单车出行绿地暴露的代码。

同时提供 OLS、随机森林、CatBoost、LightGBM 和 LSTM 作为比较模型。各模型使用独立的 Python 文件，并共用数据准备与时间验证流程。

分析流程包括：

- 模型训练与参数选择
- 模型预测性能评价
- SHAP 特征重要性分析
- 一维 SHAP 依赖与部分依赖分析
- 二维部分依赖图（PDP）分析

代码配套相关期刊论文中的研究。

---

## 研究目标

研究共享单车出行绿地暴露与时间特征、空气污染和气象条件之间的关系，具体包括：

- 预测绿地暴露及其空间分布；
- 量化时间、环境和气象变量对模型预测的贡献；
- 通过 SHAP 和 PDP 分析非线性关系及变量组合下的模型响应。

---

## 数据来源

### 共享单车出行数据

出行数据来自 Biendata 平台提供的 Mobike 竞赛数据集：

https://www.biendata.xyz/

### 绿地数据

绿地信息来自 ESA WorldCover 数据集：

https://esa-worldcover.org/en

### 空气污染数据

空气污染观测来自中国环境监测总站（CNEMC）：

https://www.cnemc.cn/

### 气象数据

气象变量来自中国气象数据服务中心（CMDC）：

https://data.cma.cn/

将匹配后的小时建模表存放在本地 `data/GE.csv`，具体列名和格式见 [数据说明](data/README.md)。

---

## 输入变量

### 预测变量

- 星期（Day of Week）
- 小时（Hour of Day）
- O3
- PM2.5
- PM10
- SO2
- NO2
- CO
- 温度（Temperature）
- 风速（Wind Speed）
- 气压（Pressure）
- 湿度（Humidity）
- 降水（Precipitation）

### 目标变量

以下四个目标分别建模：

- `SGE`：起点绿地暴露
- `EGE`：终点绿地暴露
- `GiniStart`：起点绿地暴露的基尼系数
- `GiniEnd`：终点绿地暴露的基尼系数

预测变量的列名和顺序统一在 `config.py` 中定义。

---

## 仓库结构

| 用途 | 文件 |
| --- | --- |
| 数据准备 | `prepare_data.py` |
| 六种模型 | `train_ols.py`、`train_rf.py`、`train_catboost.py`、`train_lightgbm.py`、`train_xgboost.py`、`train_lstm.py` |
| SHAP 分析 | `shap_analysis.py`、`shap_dependence.py` |
| PDP 分析 | `pdp_1d.py`、`pdp_2d.py` |
| 精度汇总与拟合图 | `summarize_results.py`、`plot_predictions.py` |
| 已有模型结果导入 | `import_saved_results.py` |
| 完整流程与检查 | `run_all.py`、`self_check.py` |
| 环境依赖 | `requirements.txt`、`requirements_lstm.txt` |

请保留共用模块和目录结构。运行生成的模型、预测、指标和图片保存在 `results/` 下。

---

## 软件环境

建议使用 **Python 3.12**。主要依赖包括 numpy、pandas、scipy、matplotlib、Pillow、scikit-learn、joblib、xgboost、lightgbm、catboost、shap 和 PyMuPDF；LSTM 使用 tensorflow 与 keras。

安装完整环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_lstm.txt
```

Windows 使用 `.venv\Scripts\activate` 激活。运行表格模型及其解释脚本可安装 `requirements.txt`，版本记录见 `environment_versions.json`。

---

## 研究方法

### 1. 模型训练

按时间排序后，将最后20%的原始记录作为测试时段，在训练期内进行五折扩展窗口交叉验证。候选配置按平均验证 MSE 选择，确定配置后用训练目标重新拟合，再评价共同测试目标。

六种模型使用相同的有效目标时刻，每个目标要求连续24小时的输入。LSTM 用当小时及此前23小时输入估计当小时目标，缩放器仅拟合训练信息，评价的是后续时段的同小时回归。OLS 和 LSTM 各使用一个配置，其他模型评价各自指定的候选参数。

### 2. 模型评价

评价指标包括 MAE、RMSE、MSE 和 R²，均从保存的预测值重新计算。实际值—预测值图使用共享坐标名称和一个图例，比较六种模型的拟合关系。

### 3. SHAP 分析

使用 SHAP 计算特征重要性、解释变量贡献，并生成总结图和依赖图。默认解释 XGBoost，也支持 OLS 和其他树模型。解释采用训练期观测，方法设置和贡献加和校验保存在结果中。

### 4. 一维 SHAP 依赖分析

SHAP 依赖图展示特征值与模型归因的关系。独立的一维 PDP 脚本展示特征网格上的平均预测。默认展示重要性前6个变量，`--top-n 13` 可分页展示全部变量。

### 5. 二维部分依赖分析

按 SHAP 重要性排序，将前三个变量分别与紧随其后的三个变量配对，形成3×3、共9个面板。

每个网格点在全部训练背景行上计算平均预测，所有面板共用真实预测范围确定的数值色标。图片输出600 dpi PNG 和可编辑文字 PDF/SVG。

SHAP 与 PDP 描述已拟合模型的行为，本身不能直接证明因果效应或交互的统计显著性。

---

## 运行代码

将本地建模表放在 `data/GE.csv`，运行完整流程：

```bash
python run_all.py --stage all
```

该流程准备共同时间划分，训练六种模型的四个目标，汇总精度，并生成 XGBoost 的 SHAP、PDP 和六模型实际值—预测值比较图。

单独训练和解释 EGE 的 XGBoost 模型：

```bash
python prepare_data.py
python train_xgboost.py --targets EGE
python shap_analysis.py --targets EGE
python shap_dependence.py --targets EGE
python pdp_1d.py --targets EGE
python pdp_2d.py --targets EGE
```

其他模型运行相应的 `train_*.py` 文件。解释脚本通过 `--model` 选择表格模型，`--results` 指定结果目录。具体选项可通过各入口的 `--help` 查看。

---

## 可重复性

输入数据格式见 [数据说明](data/README.md)。建模表由本地提供，训练生成的模型保存在 `results/models/`。输出记录数据指纹、时间划分、模型配置和逐行预测，便于核对与复现。

无需训练真实模型的检查：

```bash
python self_check.py
```

检查使用生成的测试数据，并通过 GitHub Actions 运行。匹配的已有作者模型结果可用 `import_saved_results.py` 导入。更多命令见 [完整使用说明](使用说明.txt)。

---

## 引用

在研究中使用本仓库时，请引用相关论文。

---

## 使用说明

本仓库用于学术研究和教育。使用代码时请引用相关论文。
