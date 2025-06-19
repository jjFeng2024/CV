import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_recall_curve, f1_score
)
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== Step 0. 创建输出目录 ==========
base_dir = "/Users/feng/Desktop/计算机视觉/Week4/W4_Experiment"
os.makedirs(base_dir, exist_ok=True)
existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run_")]
run_indices = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
next_run_idx = max(run_indices) + 1 if run_indices else 1
output_dir = os.path.join(base_dir, f"run_{next_run_idx}")
os.makedirs(output_dir)
print(f" 输出目录: {output_dir}")

# ========== Step 1. 加载 ORL 数据 ==========
data_dir = "/Users/feng/Desktop/计算机视觉/Week4/ORL人脸数据库"
image_size = (32, 32)
X, y = [], []
target_names = []

folders = sorted(
    [f for f in os.listdir(data_dir) if f.startswith("s") and f[1:].isdigit()],
    key=lambda f: int(f[1:])
)

for label, folder in enumerate(folders):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        target_names.append(folder)
        images = []
        for file in sorted(os.listdir(folder_path)):
            if file.lower().endswith(('.bmp', '.jpg', '.png')):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert('L').resize(image_size)
                images.append(np.array(img).flatten())
                if len(images) == 10:
                    break
        for img_flat in images:
            X.append(img_flat)
            y.append(label)

X = np.array(X)
y = np.array(y)
print(f" 加载完成，共 {len(X)} 张图像，{len(set(y))} 类别")

# ========== Step 2. 搜索超参数 ==========
split_ratios = [0.1, 0.15, 0.2, 0.25]
all_results = []
split_summary = []  # 收集每个划分比例下的测试结果
best_config = {'accuracy': 0}

for split in split_ratios:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, stratify=y, random_state=42)

    print(f"\n 当前划分比例 test_size={split}")
    print("训练集类别分布:", Counter(y_train))
    print("测试集类别分布:", Counter(y_test))

    # 自动估计最小PCA维数
    pca_temp = PCA().fit(X_train)
    cumulative_var = np.cumsum(pca_temp.explained_variance_ratio_)
    min_components = np.searchsorted(cumulative_var, 0.95) + 1

    param_grid = {
        'pca__n_components': list(range(min_components, min(min_components + 100, 200), 10)),
        'knn__n_neighbors': list(range(1, 10))
    }
    pipe = Pipeline([
        ('pca', PCA(whiten=True)),
        ('knn', KNeighborsClassifier())
    ])
    grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    for idx in range(len(grid.cv_results_['mean_test_score'])):
        all_results.append({
            'test_size': split,
            'pca_n': grid.cv_results_['param_pca__n_components'].data[idx],
            'knn_k': grid.cv_results_['param_knn__n_neighbors'].data[idx],
            'accuracy': grid.cv_results_['mean_test_score'][idx]
        })

    best_params = grid.best_params_
    pca_model = PCA(n_components=best_params['pca__n_components'], whiten=True)
    X_train_pca = pca_model.fit_transform(X_train)
    X_test_pca = pca_model.transform(X_test)
    knn_model = KNeighborsClassifier(n_neighbors=best_params['knn__n_neighbors'])
    knn_model.fit(X_train_pca, y_train)
    y_pred = knn_model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    # 记录当前划分下的测试结果到 split_summary
    split_summary.append({
        'split': split,
        'train_ratio': 1 - split,
        'accuracy': acc,
        'pca': best_params['pca__n_components'],
        'knn': best_params['knn__n_neighbors']
    })
    # 在每个 split 的末尾输出当前划分的最佳准确率
    print(f"  → 当前划分下最佳准确率为: {acc:.4f} (PCA维度={best_params['pca__n_components']}, "
          f"K={best_params['knn__n_neighbors']})")
    if acc > best_config['accuracy']:
        best_config = {
            'accuracy': acc,
            'f1': f1,
            'split': split,
            'pca': best_params['pca__n_components'],
            'knn': best_params['knn__n_neighbors'],
            'y_pred': y_pred,
            'y_test': y_test,
            'y_score': knn_model.predict_proba(X_test_pca),
            'report': classification_report(
                y_test, y_pred,
                target_names=[target_names[i] for i in np.unique(y_test)],
                zero_division=0
            ),
            'confusion': confusion_matrix(y_test, y_pred),
            'train_count': len(y_train),
            'test_count': len(y_test)
        }

# ========== Step 3. 保存结果 ==========
df_results = pd.DataFrame(all_results)
df_results.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)

# 热力图
heatmap_data = df_results[df_results['test_size'] == best_config['split']]
pivot = heatmap_data.pivot_table(index='pca_n', columns='knn_k', values='accuracy', aggfunc='mean')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt=".3f")
plt.title(f"GridSearch交叉验证平均准确率热力图\n(test_size = {best_config['split']})")
plt.xlabel("KNN邻居数")
plt.ylabel("PCA维度")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "heatmap_accuracy.png"))
plt.close()

# 分类报告
with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write("各个划分比例下的测试集准确率概览：\n")
    for record in split_summary:
        f.write(f"  - 测试集比例 {record['split']:.2f} (训练集 {record['train_ratio']:.2f})："
                f"准确率 = {record['accuracy']:.4f}，PCA维度 = {record['pca']}，KNN邻居数 = {record['knn']}\n")
    f.write("\n")

    split = best_config['split']
    train_ratio = 1 - split
    match = df_results[
        (df_results['test_size'] == split) &
        (df_results['pca_n'] == best_config['pca']) &
        (df_results['knn_k'] == best_config['knn'])
    ]
    grid_mean = match['accuracy'].values[0] if not match.empty else -1
    f.write("     最佳组合参数汇总：\n")
    f.write(f"  - 训练集与测试集比例：{int(train_ratio*10)} : {int(split*10)}（test_size = {split}）\n")
    f.write(f"  - 训练图像数：{best_config['train_count']}\n")
    f.write(f"  - 测试图像数：{best_config['test_count']}\n")
    f.write(f"  - PCA降维维数：{best_config['pca']}\n")
    f.write(f"  - KNN邻居数：{best_config['knn']}\n")
    f.write(f"  - GridSearch 平均得分：{grid_mean:.4f}\n")
    f.write(f"  - 测试集准确率：{best_config['accuracy']:.4f}\n")
    f.write(f"  - 宏平均F1分数：{best_config['f1']:.4f}\n\n")
    f.write("分类报告如下：\n")
    f.write(best_config['report'])

# 混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(best_config['confusion'], annot=True, fmt='d', cmap='Blues',
            xticklabels=[target_names[i] for i in np.unique(best_config['y_test'])],
            yticklabels=[target_names[i] for i in np.unique(best_config['y_test'])])
plt.title("混淆矩阵")
plt.xlabel("预测类别")
plt.ylabel("真实类别")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# PR 曲线
plt.figure(figsize=(8, 6))
for i in np.unique(best_config['y_test']):
    y_true_bin = (best_config['y_test'] == i).astype(int)
    y_score_bin = best_config['y_score'][:, i]
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score_bin)
    plt.plot(recall, precision, label=f"{target_names[i]}")
plt.title("PR曲线 (逐类别)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(fontsize=8, loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "PR_curves.png"))
plt.close()

# F1 曲线
plt.figure(figsize=(8, 6))
for i in np.unique(best_config['y_test']):
    y_true_bin = (best_config['y_test'] == i).astype(int)
    y_score_bin = best_config['y_score'][:, i]
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score_bin)
    f1_curve = 2 * precision * recall / (precision + recall + 1e-10)
    plt.plot(recall, f1_curve, label=f"{target_names[i]}")
plt.title("F1曲线 (逐类别)")
plt.xlabel("Recall")
plt.ylabel("F1 Score")
plt.legend(fontsize=8, loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "F1_curves.png"))
plt.close()

print(" 所有内容已保存到：", output_dir)