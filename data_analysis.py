"""
数据分析模块 - Dry Bean Dataset
功能：数据概览、类别分布、缺失值检查、异常值检测、特征分布可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DryBeanDataAnalyzer:
    """干豆数据集分析器"""
    
    # 特征名称（来自UCI数据集文档）[citation:1]
    FEATURE_NAMES = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRatio', 'Eccentricity', 'ConvexArea', 'EquivalentDiameter',
        'Extent', 'Solidity', 'Roundness', 'Compactness',
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
    ]
    CLASS_NAMES = ['Seker', 'Barbunya', 'Bombay', 'Cali', 'Dermosan', 'Horoz', 'Sira']
    
    def __init__(self, train_path, val_path, test_path):
        """初始化，加载三个数据集"""
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        
        # 合并用于整体分析
        self.full_df = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        
        print("=" * 60)
        print("干豆数据集 - 数据分析报告")
        print("=" * 60)
    
    def basic_info(self):
        """1. 数据基本信息"""
        print("\n【1. 数据基本信息】")
        print(f"总样本数: {len(self.full_df)}")
        print(f"训练集: {len(self.train_df)} 样本")
        print(f"验证集: {len(self.val_df)} 样本")
        print(f"测试集: {len(self.test_df)} 样本")
        print(f"特征数: {len(self.FEATURE_NAMES)}")
        print(f"类别数: {len(self.CLASS_NAMES)}")
        print(f"类别: {', '.join(self.CLASS_NAMES)}")
        
        # 数据来源说明
        print("\n数据来源: Koklu & Ozkan (2020), UCI Machine Learning Repository")
        print("数据用途: 通过计算机视觉和机器学习技术进行干豆品种自动分类")
        
        return self.full_df.shape
    
    def class_distribution(self):
        """2. 类别分布分析"""
        print("\n【2. 类别分布分析】")
        
        # 整体分布
        class_counts = self.full_df['Class'].value_counts()
        class_percent = (class_counts / len(self.full_df) * 100).round(2)
        
        dist_df = pd.DataFrame({
            '样本数': class_counts,
            '占比(%)': class_percent
        })
        print(dist_df)
        
        # 检查类别不平衡
        min_count = class_counts.min()
        max_count = class_counts.max()
        imbalance_ratio = max_count / min_count
        print(f"\n类别不平衡比 (最大/最小): {imbalance_ratio:.2f}")
        if imbalance_ratio > 1.5:
            print("⚠️ 存在类别不平衡问题，建议在数据处理阶段进行处理")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 柱状图
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#DDA0DD', '#F0E68C']
        axes[0].bar(class_counts.index, class_counts.values, color=colors, edgecolor='black')
        axes[0].set_title('各品种样本数量分布', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('品种')
        axes[0].set_ylabel('样本数量')
        axes[0].tick_params(axis='x', rotation=30)
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v + 5, str(v), ha='center', fontsize=10)
        
        # 饼图
        axes[1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                    colors=colors, startangle=90, explode=[0.02]*len(class_counts))
        axes[1].set_title('各品种占比分布', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ 类别分布图已保存: class_distribution.png")
        plt.show()
        
        return class_counts
    
    def missing_values_check(self):
        """3. 缺失值检查"""
        print("\n【3. 缺失值检查】")
        missing = self.full_df.isnull().sum()
        if missing.sum() == 0:
            print("✓ 数据集无缺失值，数据完整性良好")
        else:
            print(missing[missing > 0])
        return missing
    
    def outlier_detection(self):
        """4. 异常值检测（使用IQR方法）"""
        print("\n【4. 异常值检测（IQR方法）】")
        
        numeric_cols = [col for col in self.FEATURE_NAMES if col in self.full_df.columns]
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = self.full_df[col].quantile(0.25)
            Q3 = self.full_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.full_df[(self.full_df[col] < lower_bound) | 
                                    (self.full_df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
        
        outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['异常值数量'])
        outlier_df['占比(%)'] = (outlier_df['异常值数量'] / len(self.full_df) * 100).round(2)
        print(outlier_df)
        
        # 异常值总览
        total_outliers = sum(outlier_counts.values())
        print(f"\n总异常值: {total_outliers} 个样本-特征对")
        if total_outliers > 0:
            print("⚠️ 建议在数据清洗阶段使用IQR方法处理异常值")
        
        # 箱线图可视化
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        for idx, col in enumerate(numeric_cols):
            axes[idx].boxplot(self.full_df[col].dropna(), patch_artist=True,
                             boxprops=dict(facecolor='lightblue'))
            axes[idx].set_title(col, fontsize=9)
            axes[idx].set_ylabel('值')
        for idx in range(len(numeric_cols), 16):
            axes[idx].axis('off')
        plt.suptitle('各特征箱线图 - 异常值检测', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outlier_boxplots.png', dpi=300, bbox_inches='tight')
        print("\n✓ 箱线图已保存: outlier_boxplots.png")
        plt.show()
        
        return outlier_counts
    
    def feature_distribution(self):
        """5. 特征分布分析"""
        print("\n【5. 特征分布分析】")
        
        numeric_cols = [col for col in self.FEATURE_NAMES if col in self.full_df.columns]
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        for idx, col in enumerate(numeric_cols):
            axes[idx].hist(self.full_df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(col, fontsize=9)
            axes[idx].set_xlabel('值')
            axes[idx].set_ylabel('频数')
        for idx in range(len(numeric_cols), 16):
            axes[idx].axis('off')
        plt.suptitle('各特征分布直方图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_distribution.png', dpi=300, bbox_inches='tight')
        print("✓ 特征分布图已保存: feature_distribution.png")
        plt.show()
    
    def correlation_analysis(self):
        """6. 特征相关性分析"""
        print("\n【6. 特征相关性分析】")
        
        numeric_cols = [col for col in self.FEATURE_NAMES if col in self.full_df.columns]
        corr_matrix = self.full_df[numeric_cols].corr()
        
        # 找出高相关特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print("⚠️ 存在高度相关特征 (|r| > 0.9):")
            for pair in high_corr_pairs:
                print(f"  {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
            print("建议: 考虑特征选择或PCA降维以减少冗余")
        else:
            print("✓ 未发现高度相关特征对")
        
        # 热力图
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={'shrink': 0.8})
        ax.set_title('特征相关性热力图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ 相关性热力图已保存: correlation_heatmap.png")
        plt.show()
        
        return corr_matrix
    
    def run_full_analysis(self):
        """运行完整数据分析"""
        self.basic_info()
        self.missing_values_check()
        self.class_distribution()
        self.outlier_detection()
        self.feature_distribution()
        self.correlation_analysis()
        print("\n" + "=" * 60)
        print("数据分析完成！所有图表已保存。")
        print("=" * 60)


if __name__ == "__main__":
    # 使用示例
    analyzer = DryBeanDataAnalyzer(
        train_path='../data/train.csv',
        val_path='../data/val.csv',
        test_path='../data/test.csv'
    )
    analyzer.run_full_analysis()
