"""
多算法实验分析模块 - Dry Bean Dataset
实现: Random Forest, XGBoost (课堂未讲), SVM, 以及鲁棒性分析
参考论文: XGBoost在干豆数据集上可达92.88%准确率 [citation:2]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DryBeanExperiment:
    """干豆分类多算法实验分析"""
    
    def __init__(self, data, label_encoder):
        """
        初始化实验
        data: get_processed_data() 返回的字典
        """
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.label_encoder = label_encoder
        
        self.models = {}
        self.results = {}
        self.train_times = {}
        self.inference_times = {}
        self.history = {}  # 用于存储XGBoost训练历史
        
        print("=" * 60)
        print("干豆数据集 - 多算法实验分析")
        print("=" * 60)
        print(f"训练集: {self.X_train.shape}")
        print(f"验证集: {self.X_val.shape}")
        print(f"测试集: {self.X_test.shape}")
    
    def train_random_forest(self, n_estimators=100, max_depth=20, random_state=42):
        """训练随机森林模型 [citation:2]"""
        print("\n【训练 Random Forest】")
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        start = time.time()
        model.fit(self.X_train, self.y_train)
        self.train_times['Random Forest'] = time.time() - start
        
        self.models['Random Forest'] = model
        print(f"✓ 训练完成，耗时: {self.train_times['Random Forest']:.2f}秒")
        return model
    
    def train_xgboost(self, n_estimators=600, max_depth=4, learning_rate=0.15,
                      random_state=42, eval_set_ratio=0.2):
        """
        训练XGBoost模型（课堂未讲的算法）
        参数参考自干豆数据集最优配置 [citation:2]
        """
        print("\n【训练 XGBoost】（课堂未讲算法）")
        
        # 从训练集划分部分作为早停验证集
        split_idx = int(len(self.X_train) * (1 - eval_set_ratio))
        X_train_early = self.X_train[:split_idx]
        y_train_early = self.y_train[:split_idx]
        X_eval = self.X_train[split_idx:]
        y_eval = self.y_train[split_idx:]
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        start = time.time()
        model.fit(
            X_train_early, y_train_early,
            eval_set=[(X_train_early, y_train_early), (X_eval, y_eval)],
            verbose=False
        )
        self.train_times['XGBoost'] = time.time() - start
        
        # 获取训练历史
        self.history['XGBoost'] = model.evals_result()
        
        self.models['XGBoost'] = model
        print(f"✓ 训练完成，耗时: {self.train_times['XGBoost']:.2f}秒")
        return model
    
    def train_svm(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        """训练SVM模型"""
        print("\n【训练 SVM】")
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True  # 用于后续鲁棒性分析
        )
        
        start = time.time()
        model.fit(self.X_train, self.y_train)
        self.train_times['SVM'] = time.time() - start
        
        self.models['SVM'] = model
        print(f"✓ 训练完成，耗时: {self.train_times['SVM']:.2f}秒")
        return model
    
    def train_all(self):
        """训练所有算法"""
        self.train_random_forest()
        self.train_xgboost()
        self.train_svm()
        print("\n所有算法训练完成！")
    
    def evaluate_model(self, model, X, y_true, model_name):
        """评估单个模型"""
        start = time.time()
        y_pred = model.predict(X)
        inference_time = time.time() - start
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted'),
            'Recall': recall_score(y_true, y_pred, average='weighted'),
            'F1 Score': f1_score(y_true, y_pred, average='weighted'),
            'Inference Time (s)': inference_time / len(X)  # 平均每样本耗时
        }
        
        return metrics, y_pred
    
    def evaluate_all(self):
        """评估所有算法"""
        print("\n【模型评估】")
        
        for name, model in self.models.items():
            print(f"\n--- {name} ---")
            
            # 验证集评估
            val_metrics, _ = self.evaluate_model(model, self.X_val, self.y_val, name)
            # 测试集评估
            test_metrics, y_pred = self.evaluate_model(model, self.X_test, self.y_test, name)
            
            self.results[name] = {
                'val': val_metrics,
                'test': test_metrics,
                'y_pred': y_pred
            }
            
            print(f"验证集 - Accuracy: {val_metrics['Accuracy']:.4f}, F1: {val_metrics['F1 Score']:.4f}")
            print(f"测试集 - Accuracy: {test_metrics['Accuracy']:.4f}, F1: {test_metrics['F1 Score']:.4f}")
        
        return self.results
    
    def plot_loss_curves(self):
        """绘制XGBoost的Loss曲线（非训练型算法不做）[citation:2]"""
        if 'XGBoost' not in self.history:
            print("⚠️ 无XGBoost训练历史，跳过Loss曲线绘制")
            return
        
        print("\n【绘制Loss曲线】")
        evals = self.history['XGBoost']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(evals['validation_0']['mlogloss'], label='训练集', linewidth=2)
        ax.plot(evals['validation_1']['mlogloss'], label='验证集', linewidth=2)
        ax.set_xlabel('迭代轮次', fontsize=12)
        ax.set_ylabel('多分类交叉熵损失 (mlogloss)', fontsize=12)
        ax.set_title('XGBoost 训练损失曲线', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('xgboost_loss_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Loss曲线已保存: xgboost_loss_curves.png")
        plt.show()
    
    def plot_accuracy_comparison(self):
        """绘制精度对比柱状图"""
        print("\n【精度对比】")
        
        names = list(self.results.keys())
        test_acc = [self.results[n]['test']['Accuracy'] for n in names]
        val_acc = [self.results[n]['val']['Accuracy'] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, val_acc, width, label='验证集', color='skyblue')
        bars2 = ax.bar(x + width/2, test_acc, width, label='测试集', color='coral')
        
        ax.set_xlabel('算法', fontsize=12)
        ax.set_ylabel('准确率', fontsize=12)
        ax.set_title('各算法精度对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        ax.set_ylim(0.7, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 精度对比图已保存: accuracy_comparison.png")
        plt.show()
    
    def speed_comparison(self):
        """推理速度对比"""
        print("\n【推理速度对比】")
        
        speed_df = pd.DataFrame({
            '算法': list(self.results.keys()),
            '平均推理时间 (秒/样本)': [self.results[n]['test']['Inference Time (s)'] for n in self.results.keys()],
            '训练时间 (秒)': [self.train_times[n] for n in self.results.keys()]
        })
        speed_df['推理速度 (样本/秒)'] = 1 / speed_df['平均推理时间 (秒/样本)']
        print(speed_df.to_string(index=False))
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(speed_df['算法'], speed_df['推理速度 (样本/秒)'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax.set_xlabel('算法', fontsize=12)
        ax.set_ylabel('推理速度 (样本/秒)', fontsize=12)
        ax.set_title('各算法推理速度对比', fontsize=14, fontweight='bold')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig('speed_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ 速度对比图已保存: speed_comparison.png")
        plt.show()
        
        return speed_df
    
    def robustness_test(self, noise_types=None, noise_levels=None):
        """
        鲁棒性测试：在测试数据上添加不同噪声，观察精度下降 [citation:2]
        """
        if noise_types is None:
            noise_types = ['gaussian', 'laplace']
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.2]
        
        print("\n【鲁棒性分析】")
        print("在测试数据上添加不同噪声，观察各算法精度下降情况")
        
        robustness_results = {}
        
        for noise_type in noise_types:
            for noise_level in noise_levels:
                # 生成噪声
                noise = self._generate_noise(self.X_test, noise_type, noise_level)
                X_noisy = self.X_test + noise
                
                key = f"{noise_type}_std{noise_level}"
                robustness_results[key] = {}
                
                for name, model in self.models.items():
                    y_pred = model.predict(X_noisy)
                    acc = accuracy_score(self.y_test, y_pred)
                    robustness_results[key][name] = acc
        
        # 整理结果
        results_df = pd.DataFrame(robustness_results).T
        results_df = results_df[self.models
