"""
数据处理模块 - Dry Bean Dataset
功能：数据清洗（异常值处理）、特征工程（标准化、降维）、类别不平衡处理
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
import warnings
warnings.filterwarnings('ignore')


class DryBeanDataProcessor:
    """干豆数据集处理器"""
    
    FEATURE_NAMES = [
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
        'AspectRatio', 'Eccentricity', 'ConvexArea', 'EquivalentDiameter',
        'Extent', 'Solidity', 'Roundness', 'Compactness',
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
    ]
    
    def __init__(self, train_path, val_path, test_path, random_state=42):
        """初始化，加载数据"""
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)
        self.random_state = random_state
        
        # 分离特征和标签
        self.X_train_raw = self.train_df[self.FEATURE_NAMES].copy()
        self.y_train_raw = self.train_df['Class'].copy()
        self.X_val_raw = self.val_df[self.FEATURE_NAMES].copy()
        self.y_val_raw = self.val_df['Class'].copy()
        self.X_test_raw = self.test_df[self.FEATURE_NAMES].copy()
        self.y_test_raw = self.test_df['Class'].copy()
        
        # 编码标签
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train_raw)
        self.y_val_encoded = self.label_encoder.transform(self.y_val_raw)
        self.y_test_encoded = self.label_encoder.transform(self.y_test_raw)
        
        print("=" * 60)
        print("干豆数据集 - 数据处理报告")
        print("=" * 60)
        print(f"原始训练集: {len(self.X_train_raw)} 样本, {len(self.FEATURE_NAMES)} 特征")
        print(f"原始验证集: {len(self.X_val_raw)} 样本")
        print(f"原始测试集: {len(self.X_test_raw)} 样本")
    
    def outlier_removal_iqr(self, df, multiplier=1.5):
        """
        使用IQR方法去除异常值
        IQR = Q3 - Q1，异常值定义为 < Q1 - 1.5*IQR 或 > Q3 + 1.5*IQR [citation:1]
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # 保留在边界内的样本
        mask = ((df >= lower_bound) & (df <= upper_bound)).all(axis=1)
        return df[mask], mask
    
    def clean_data(self, method='iqr'):
        """
        数据清洗主函数
        method: 'iqr' 使用IQR去除异常值
        """
        print("\n【数据清洗】")
        
        # 1. 缺失值检查（数据无缺失，但做验证）
        if self.X_train_raw.isnull().sum().sum() == 0:
            print("✓ 训练集无缺失值")
        
        # 2. 异常值处理 - 仅对训练集进行，避免数据泄露
        if method == 'iqr':
            X_clean, clean_mask = self.outlier_removal_iqr(self.X_train_raw)
            y_clean = self.y_train_encoded[clean_mask]
            
            removed_count = len(self.X_train_raw) - len(X_clean)
            print(f"✓ IQR方法去除异常值: 移除 {removed_count} 个样本 ({removed_count/len(self.X_train_raw)*100:.2f}%)")
            
            self.X_train = X_clean
            self.y_train = y_clean
        else:
            self.X_train = self.X_train_raw
            self.y_train = self.y_train_encoded
        
        # 验证集和测试集保留原样（不做异常值移除，模拟真实场景）
        self.X_val = self.X_val_raw
        self.y_val = self.y_val_encoded
        self.X_test = self.X_test_raw
        self.y_test = self.y_test_encoded
        
        return self.X_train, self.y_train
    
    def handle_imbalance(self, method='smote_tomek'):
        """
        处理类别不平衡问题 [citation:2]
        method: 'smote_tomek' 或 'adasyn'
        """
        print("\n【类别不平衡处理】")
        
        # 检查原始类别分布
        unique, counts = np.unique(self.y_train, return_counts=True)
        print(f"原始类别分布: {dict(zip(self.label_encoder.classes_[unique], counts))}")
        
        if method == 'smote_tomek':
            # SMOTE + Tomek Links 组合方法 [citation:2]
            sampler = SMOTETomek(random_state=self.random_state)
            self.X_train_resampled, self.y_train_resampled = sampler.fit_resample(
                self.X_train, self.y_train
            )
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.random_state)
            self.X_train_resampled, self.y_train_resampled = sampler.fit_resample(
                self.X_train, self.y_train
            )
        else:  # SMOTE
            sampler = SMOTE(random_state=self.random_state)
            self.X_train_resampled, self.y_train_resampled = sampler.fit_resample(
                self.X_train, self.y_train
            )
        
        unique_resampled, counts_resampled = np.unique(self.y_train_resampled, return_counts=True)
        print(f"平衡后类别分布: {dict(zip(self.label_encoder.classes_[unique_resampled], counts_resampled))}")
        print(f"✓ 使用 {method} 方法，样本数从 {len(self.X_train)} 增至 {len(self.X_train_resampled)}")
        
        return self.X_train_resampled, self.y_train_resampled
    
    def scale_features(self, method='standard'):
        """
        特征缩放
        method: 'standard' 标准化 或 'minmax' 归一化
        """
        print("\n【特征缩放】")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        # 使用训练集拟合，然后转换所有数据集
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_resampled)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"✓ 使用 {method} 缩放，特征均值/方差已标准化")
        print(f"  训练集: {self.X_train_scaled.shape}")
        print(f"  验证集: {self.X_val_scaled.shape}")
        print(f"  测试集: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_val_scaled, self.X_test_scaled
    
    def apply_pca(self, n_components=0.95):
        """
        PCA降维（可选）
        n_components: 保留的方差比例或主成分数量
        """
        print("\n【PCA降维】")
        
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.X_train_pca = self.pca.fit_transform(self.X_train_scaled)
        self.X_val_pca = self.pca.transform(self.X_val_scaled)
        self.X_test_pca = self.pca.transform(self.X_test_scaled)
        
        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"✓ PCA保留 {explained_variance*100:.2f}% 方差")
        print(f"  特征数: {self.X_train_scaled.shape[1]} → {self.X_train_pca.shape[1]}")
        
        return self.X_train_pca, self.X_val_pca, self.X_test_pca
    
    def get_processed_data(self, clean_method='iqr', imbalance_method='smote_tomek',
                          scale_method='standard', use_pca=False, pca_components=0.95):
        """
        一站式数据处理流水线
        """
        print("\n" + "=" * 60)
        print("开始数据处理流水线")
        print("=" * 60)
        
        # Step 1: 数据清洗
        self.clean_data(method=clean_method)
        
        # Step 2: 类别不平衡处理
        self.handle_imbalance(method=imbalance_method)
        
        # Step 3: 特征缩放
        self.scale_features(method=scale_method)
        
        # Step 4: PCA降维（可选）
        if use_pca:
            self.apply_pca(n_components=pca_components)
            X_train_final = self.X_train_pca
            X_val_final = self.X_val_pca
            X_test_final = self.X_test_pca
        else:
            X_train_final = self.X_train_scaled
            X_val_final = self.X_val_scaled
            X_test_final = self.X_test_scaled
        
        print("\n" + "=" * 60)
        print("数据处理完成！")
        print(f"最终训练集: {X_train_final.shape}")
        print(f"最终验证集: {X_val_final.shape}")
        print(f"最终测试集: {X_test_final.shape}")
        print("=" * 60)
        
        return {
            'X_train': X_train_final,
            'y_train': self.y_train_resampled,
            'X_val': X_val_final,
            'y_val': self.y_val,
            'X_test': X_test_final,
            'y_test': self.y_test,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'pca': self.pca if use_pca else None
        }


if __name__ == "__main__":
    # 使用示例
    processor = DryBeanDataProcessor(
        train_path='../data/train.csv',
        val_path='../data/val.csv',
        test_path='../data/test.csv'
    )
    
    data = processor.get_processed_data(
        clean_method='iqr',
        imbalance_method='smote_tomek',
        scale_method='standard',
        use_pca=False
    )
    
    print(f"\n数据准备就绪: X_train {data['X_train'].shape}, y_train {data['y_train'].shape}")
