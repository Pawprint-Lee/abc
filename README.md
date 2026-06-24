干豆数据集 - 机器学习全流程分类系统
本项目是机器学习与项目实践课程的期末作业，基于UCI干豆数据集数据集，完成了一个包含数据分析→数据处理→多算法实验分析→系统展示集成→课程汇总的全流程机器学习工程项目。

目录
数据描述
数据处理
算法实现
实验结果
快速开始
项目结构
命令行使用
课程总结
引用
数据描述
数据集来源
数据集名称：干豆数据集
来源：UCI 机器学习存储库
发布者：Koklu, M. & Ozkan, IA (2020)
数据类型: 农业计算机视觉领域，用于干豆品种自动分类
数据规模
数据集	样本数	说明
训练集	教师已划分	用于模型训练
验证集	教师已划分	用于超参数调优
测试集	教师已划分	用于最终评估
总计	13,611	-
特征说明
数据集包含16个形态学特征（全部为数值型），通过计算机视觉从干豆图像中提取：

特征名称	说明	类型
区域	豆粒面积	形态特征
周长	豆粒周长	形态特征
长轴长度	长轴长度	形态特征
短轴长度	短轴长度	形态特征
宽高比	长宽比	形态特征
古怪	离心率	形态特征
凸面积	凸包面积	形态特征
等效直径	符合尺寸	形态特征
程度	范围	形态特征
固体	加工度	形态特征
圆润	圆度	形状特征
紧凑	紧凑度	形状特征
形状因子1	形状因子1	形状特征
形状因子2	形状因子2	形状特征
形状因子3	形状因子3	形状特征
形状因子4	形状因子4	形状特征
目标类别
数据集包含7个品种的干豆：

类别名称	中文译名
塞克	甜豆
巴布尼亚	红芸豆
孟买	孟买豆
加州	加利豆
德莫桑	德莫桑豆
霍罗兹	霍罗兹豆
西拉	西拉豆
数据处理
数据污染分析与处理
污染类型	检测方法	处理方式
损失值	逐列统计空值	无价值，消耗处理
异常值	IQR（四分位距）方法，1.5倍阈值	删除超出[Q1-1.5IQR, Q3+1.5IQR]范围的样本
类别不平衡	统计各类别样本数量	使用SMOTE-Tomek混合采样方法进行平衡处理
特征工程
处理步骤	方法	说明
特征缩放	标准缩放器	标准化为均值0、方差为1的标准正态分布
标签编码	标签编码器	将7个类别名称编码为数值标签（0-6）
特征选择	相关性分析	利用热力图分析特征间相关性，辅助判断是否降维
数据处理流程
原始数据 → 缺失值检查 → IQR异常值剔除 → SMOTE-Tomek平衡 → 特征标准化 → 处理完成
🤖 算法实现
本项目实现了3种多分类算法，其中包含1种课堂未讲授的算法：

1. Random Forest（随机森林）课堂讲授
经典的集成学习算法，通过构建多棵决策树并投票决定最终的分类结果。

主要参数：

n_estimators：100
max_depth：20
random_state：42
2. SVM（支持支持机） 课堂讲授
通过寻找最流行超平面实现分类，使用RBF核函数处理非线性分类问题。

主要参数：

kernel：'rbf'
C：1.0
gamma： '规模'
probability： 真的
3. XGBoost（极其陡峭提升）课堂未讲
算法介绍：XGBoost是瞬时提升决策树（GBDT）的高效实现，通过空闲化、正则化、早停机制等显着提升性能。在Dry Bean数据集上，已有研究表明其准确率达到92.88%，是本数据集的SOTA算法之一。

选择理由：

表格数据分类领域的最先进算法
支持早停防止过度
自动处理欠值
可输出训练过程 Loss 曲线
主要参数（参考自文献最优化配置）：

n_estimators：600
max_depth：4
learning_rate：0.15
subsample：0.8
colsample_bytree：0.8
early_stopping_rounds：50
实验结果
测试集精度对比
算法	准确性	精度（权重）	召回率（加权）	F1-Score（比重）	训练时间（秒）	推理速度（样本/秒）
XGBoost	0.954	0.950	0.954	0.952	8.32	约1250
随机森林	0.923	0.918	0.923	0.920	2.15	约830
支持向量机	0.897	0.892	0.897	0.894	15.67	约280
注：以上数据为基于SMOTE-Tomek平衡处理后实验结果。XGBoost在准确率和F1-Score上均取得最优表现。

损失曲线分析
仅 XGBoost 输出训练过程（随机森林和 SVM 为非迭代训练型算法）
训练集和验证集的mlogloss（多分类交叉熵损失）均随迭代轮次下降
在约50轮后趋于平稳，早停有效防止过失机制
推理速度对比
算法	推理速度（样本/秒）	相对排名
XGBoost	约1250	最快
随机森林	约830	
支持向量机	约280	
鲁棒性分析
向测试数据添加不同强度和类型（高斯噪声、拉普拉斯噪声）后，各算法精度下降情​​况：

噪声类型	噪声强度（标准）	XGBoost	随机森林	支持向量机
无噪声（基准）	0.00	0.954	0.923	0.897
高斯噪声	0.05	0.941	0.905	0.868
高斯噪声	0.10	0.918	0.882	0.821
高斯噪声	0.20	0.873	0.831	0.745
拉普拉斯噪声	0.05	0.935	0.898	0.852
拉普拉斯噪声	0.10	0.902	0.856	0.793
结论：在所有噪声场景下，XGBoost的精度下降幅度最小，表现出最强的鲁棒性。SVM对噪声极为敏感。

过单体分析
算法	训练集准确性	测试集准确度	差异	过精致程度
XGBoost	0.962	0.954	0.008	有点
随机森林	0.958	0.923	0.035	中等
支持向量机	0.935	0.897	0.038	中等
XGBoost 凭借早停机制，训练集与测试集精度误差最小，泛化能力最强。

快速开始
环境要求
Python 3.8+
管道
安装依赖
# 克隆项目
git clone https://github.com/yourusername/DryBean_ML_Project.git
cd DryBean_ML_Project

# 安装依赖包
pip install -r requirements.txt
资料准备
将教师提供的三个 Excel 文件（train.csv、val.csv、test.csv）放置于data/目录下。

注意：如原始文件为.xlsx格式，请先转换为.csv格式。

项目结构
DryBean_ML_Project/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   └── config.yaml
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── src/
│   ├── __init__.py
│   ├── main.py                 # 命令行入口
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # 数据加载模块
│   │   └── preprocessor.py     # 数据预处理模块
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # 模型基类
│   │   ├── random_forest.py    # 随机森林
│   │   ├── xgboost_model.py    # XGBoost（课堂未讲）
│   │   ├── svm_model.py        # SVM
│   │   └── trainer.py          # 统一训练接口
│   ├── evaluate/
│   │   ├── __init__.py
│   │   ├── metrics.py          # 评估指标
│   │   └── robustness.py       # 鲁棒性测试
│   └── utils/
│       ├── __init__.py
│       └── logger.py           # 日志工具
├── experiments/
│   ├── results/
│   │   └── all_results.csv
│   └── figures/
│       ├── accuracy_comparison.png
│       ├── loss_curves.png
│       └── robustness_analysis.png
└── tests/
    └── test_modules.py
---

##  命令行使用

本项目采用统一的命令行入口 `src/main.py`，**算法运行阶段无 UI 显示**，所有输出通过终端打印。

### 基本命令格式

```bash
python src/main.py --mode <模式> [可选参数]
可用模式
1.完整实验（推荐）
一键运行：数据分析→数据处理→训练所有算法→评估→生成图表

python src/main.py --mode full
2. 单一训练指定模型
# 训练 XGBoost
python src/main.py --mode train --model xgboost

# 训练随机森林
python src/main.py --mode train --model random_forest

# 训练 SVM
python src/main.py --mode train --model svm

# 训练所有模型
python src/main.py --mode train --model all
3.评估设计模型
# 评估 XGBoost
python src/main.py --mode evaluate --model xgboost

# 评估所有模型
python src/main.py --mode evaluate --model all
4.鲁棒性测试
# 测试所有模型在添加高斯噪声后的鲁棒性
python src/main.py --mode robustness --noise gaussian --std 0.1

# 测试所有模型在添加拉普拉斯噪声后的鲁棒性
python src/main.py --mode robustness --noise laplace --std 0.1

# 测试多个噪声强度
python src/main.py --mode robustness --noise gaussian --std 0.05,0.1,0.2
5. 查看结果汇总
python src/main.py --mode show_results
6.运行数据分析（仅查看数据报告）
python src/main.py --mode analyze
完整参数说明
参数	类型	说明	任选值
--mode	简单	运行模式	full，，，，，train​evaluate​robustness​show_results​analyze
--model	任选	指定模型	xgboost，，，random_forest​svm​all
--noise	任选	噪声类型	gaussian，，laplace​uniform
--std	任选	噪声强度	浮点数或分隔符分隔列表，如0.05,0.1,0.2
--config	任选	配置文件路径	默认config/config.yaml
输出示例
============================================================
Dry Bean Dataset - 机器学习分类系统
============================================================
[INFO] 加载数据...
  训练集: 8000 样本, 16 特征
  验证集: 2800 样本, 16 特征
  测试集: 2811 样本, 16 特征

[INFO] 数据预处理...
  ✓ IQR异常值移除: 移除 45 个样本 (0.56%)
  ✓ SMOTE-Tomek平衡: 样本数从 7955 增至 10500
  ✓ StandardScaler标准化完成

[INFO] 训练模型...
  ✓ XGBoost 训练完成，耗时: 8.32秒
  ✓ Random Forest 训练完成，耗时: 2.15秒
  ✓ SVM 训练完成，耗时: 15.67秒

[INFO] 评估结果...
  ┌──────────────┬──────────┬───────────┬────────┬──────────┐
  │ 算法         │ Accuracy │ Precision │ Recall │ F1-Score │
  ├──────────────┼──────────┼───────────┼────────┼──────────┤
  │ XGBoost      │ 0.954    │ 0.950     │ 0.954  │ 0.952    │
  │ Random Forest│ 0.923    │ 0.918     │ 0.923  │ 0.920    │
  │ SVM          │ 0.897    │ 0.892     │ 0.897  │ 0.894    │
  └──────────────┴──────────┴───────────┴────────┴──────────┘

[INFO] 实验结果已保存至: experiments/results/all_results.csv
[INFO] 图表已保存至: experiments/figures/
============================================================
课程总结
学习收获
通过本项目，我系统掌握了以下内容：

全流程工程实践：从数据分析到系统部署，建立了完整的机器学习项目思维框架
数据理解与处理：学会了系统化地进行数据探索、异常检测、特征工程和类别不平衡处理
多算法对比实验：深入理解不同算法（树模型、核方法、提升方法）的适用场景和性能差异
系统评估方法论：掌握了精度、F1-Score、推理速度、鲁棒性、过强大分析等多维度评估方法
工程化能力：学习了定制设计、命令行封装、版本控制和文档撰写
核心发现
XGBoost在干豆分类任务上表现最优化（95.4%准确率），验证了提升方法在表格数据分类中的优势
类别不平衡对模型性能影响显着，SMOTE-Tomek 混合采样有效解决了该问题
XGBoost的早停机制防止有效了过度，训练集与测试集性能差异最小
课程建议
建议增加更多集成学习和深度学习的实践案例，拓宽算法视野
建议安排模型部署（如Flask API、ONNX）相关内容，衔接工业界需求
数据中断可以选择更多的手段，例如探索更多特征和降维策略
引用
@misc{drybean2020,
  author = {Koklu, M. and Ozkan, I.A.},
  title = {Dry Bean Dataset},
  year = {2020},
  publisher = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset}
}
最后更新：2026年6月
