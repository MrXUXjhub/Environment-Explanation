# 绿地暴露的时间序列建模与可解释分析

[English](README.md) · [完整中文使用说明](使用说明.txt)

本仓库将 OLS、随机森林、CatBoost、LightGBM、XGBoost、LSTM 拆成独立训练文件，
分别提供 SHAP 总结图、SHAP 依赖图、一维 PDP 和二维 PDP。
六种模型均剔除 **trip volume 和 travel distance**，使用相同的13个输入变量、
目标时刻和时间验证划分。原始单文件代码原样保存在 legacy/。

## 文件与数据

| 用途 | 文件 |
| --- | --- |
| 六种模型 | train_ols.py、train_rf.py、train_catboost.py、train_lightgbm.py、train_xgboost.py、train_lstm.py |
| SHAP | shap_analysis.py、shap_dependence.py |
| PDP | pdp_1d.py、pdp_2d.py |
| 数据与检查 | prepare_data.py、self_check.py |
| 指标与拟合图 | summarize_results.py、plot_predictions.py |
| 已有结果导入 | import_saved_results.py |
| 分阶段完整运行 | run_all.py |

保留全部共用模块和目录结构。13个输入为星期、小时、O₃、PM₂.₅、PM₁₀、SO₂、
NO₂、CO、温度、风速、气压、湿度、降水，列名和顺序见 config.py。
四个目标为 SGE、EGE、GiniStart、GiniEnd，分别训练和报告。

本次公开更新不包含观测 CSV、训练权重及生成结果。
将本地小时表放在 data/GE.csv，格式见 [数据说明](data/README.md)。
CSV 中即使保留订单数和距离列，模型也只从13列白名单构造输入。

## 安装与运行

建议 Python 3.12，在仓库根目录执行：

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_lstm.txt
~~~

Windows 激活命令为 .venv\Scripts\activate。
不运行 LSTM 时可安装 requirements.txt。

先执行无需真实模型训练的检查，再准备本地数据并运行单个目标：

~~~bash
python self_check.py
python prepare_data.py
python train_xgboost.py --targets EGE
python shap_analysis.py --targets EGE
python shap_dependence.py --targets EGE
python pdp_1d.py --targets EGE
python pdp_2d.py --targets EGE
~~~

完整训练六种模型、四个目标并生成解释图：

~~~bash
python run_all.py --stage all
~~~

该命令会实际训练。可单独选择 prepare、check、fit、summary、explain、plots 阶段。
自定义 CSV 使用 prepare_data.py --data /path/to/GE.csv；
自定义结果目录使用 --results，相关后续命令必须指向同一目录。

self_check.py 不需要真实 CSV，主要检查使用明确标注的合成测试数据。
本地存在指纹一致的作者 CSV 时，才运行额外的原数据回归检查。
GitHub Actions 执行无需数据的检查，不重新训练六种模型。

## 时间序列验证

按小时排序后，前80%原始记录为训练时段，后20%为共同参考测试时段。
每个目标时刻必须有连续24小时输入，不能跨越缺失日期。
该筛选也应用于表格模型，确保六模型比较相同的目标时刻。

训练期使用五折扩展窗口，每折用过去训练、后续时间块验证，gap=0。
候选配置按平均验证 MSE 选择，不按测试精度选择。
确定配置后，在全部训练目标上重拟合，再评价共同测试时段。

原数据336条记录对应222个训练目标、68个测试目标；每折验证37个目标，
训练目标依次为37、74、111、148、185个。
缺失小时意味着等样本数不等于等日历跨度，以上数量仅适用于原数据。
该参考测试时段此前已用于15变量比较，并非新增外部验证集。

LSTM 输入 X[t-23] … X[t]、输出 y[t]，是后续时段的同小时回归，并非提前24小时预测。
每折缩放器只拟合训练信息，固定50轮，不用最终测试集早停。
OLS 和 LSTM 均只有一个配置，不描述为多配置搜索。
默认 XGBoost 保留原三组候选参数，扩展搜索另名保存。
指标按真实预测计算，保留负 R²，不强制任何模型排名第一。

## SHAP、PDP 与图形

默认解释 XGBoost，也支持 OLS 和其他树模型；
不把表格 TreeSHAP 直接用于 LSTM。
全部解释来自训练期，方法、背景设置和加和误差保存到元数据。
总结图展示13个变量；依赖图和一维 PDP 默认前6个，
--top-n 13 可分页展示全部。

二维 PDP 为3×3布局：按训练期平均绝对 SHAP 排序，前三个变量分别与紧随其后的
三个变量配对，即第1名与第2—4名、第2名与第3—5名、第3名与第4—6名。
每个网格点都替换全部训练背景行中的对应输入，再平均模型预测。
九个子图使用按真实预测范围确定的统一数值色标。

图形输出600 dpi PNG、可编辑文字 PDF/SVG，检查面板对齐、字体和文字重叠。
六模型实际值—预测值图采用共享坐标名称和一个图例。
SHAP 是模型归因，PDP 是平均模型预测，均不能直接证明因果效应或交互显著性。
原普通独立 bootstrap 和 LOWESS 置信带没有沿用。

## 不重新训练时复用已有13变量结果

~~~bash
python prepare_data.py
python import_saved_results.py --source-results /path/to/paper5_no_mobility
python shap_analysis.py --targets EGE
python pdp_2d.py --targets EGE
~~~

导入器仅适用于此前作者结果格式，核对数据指纹、变量、时间索引及表格模型预测。
原15变量模型不能通过删图或改标签变成13变量模型。

本次仓库整理没有重新训练六种模型，也不产生新的论文精度结论。
实际检查范围见 [validation_report.json](validation_report.json)。
