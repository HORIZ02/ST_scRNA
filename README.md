#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scanpy as sc  # 单细胞分析核心库
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
import scipy.sparse
import math

# 过滤警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


# ================================
# 全局样式设置 - 罗马字体和无网格
# ================================
def set_roman_font_no_grid():
    """设置罗马字体和无网格样式"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'grid.linewidth': 0,
        'grid.alpha': 0,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.format': 'jpg',
        'figure.dpi': 600,
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 32
    })

    # 设置seaborn样式
    sns.set_style("white", {
        'axes.grid': False,
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': 'none'
    })


# 应用全局样式
set_roman_font_no_grid()

# ================================
# 路径设置
# ================================
base_dir = r"D:\Data_set\"
matrix_file = os.path.join(base_dir, "")
barcodes_file = os.path.join(base_dir, "")
genes_file = os.path.join(base_dir, "")
metadata_file = os.path.join(base_dir, "")
out_dir = os.path.join(base_dir, "")
os.makedirs(out_dir, exist_ok=True)

# 设置scanpy参数
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=600, facecolor='white', figsize=(15, 12))
sc.settings.figdir = out_dir

print(f"分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ================================
# 配置类
# ================================
class Config:
    # QC参数
    MIN_GENES = 200
    MAX_MT_PERCENT = 20
    MIN_CELLS = 3

    # 归一化参数
    TARGET_SUM = 1e4

    # 高变基因参数
    N_TOP_GENES = 2000
    HVG_MIN_MEAN = 0.0125
    HVG_MAX_MEAN = 3
    HVG_MIN_DISP = 0.5

    # 聚类参数
    RESOLUTIONS = [0.3, 0.5, 0.8]
    N_NEIGHBORS = 15
    N_PCS = 40


config = Config()


# ================================
# 工具函数
# ================================
def optimize_memory_usage(adata):
    """优化AnnData对象的内存使用"""
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    elif not isinstance(adata.X, scipy.sparse.csr_matrix):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            try:
                if adata.obs[col].nunique() / len(adata.obs[col]) < 0.5:
                    adata.obs[col] = adata.obs[col].astype('category')
            except:
                pass

    return adata


def save_figure_multiformat(fig, filename_base, out_dir, formats=['jpg', 'pdf']):
    """以多种格式保存图像"""
    for fmt in formats:
        filepath = os.path.join(out_dir, f"{filename_base}.{fmt}")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"保存图像: {filepath}")


def create_clean_axis(ax):
    """创建干净的坐标轴，无网格线"""
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    return ax


def add_cluster_labels(ax, adata, embedding='X_umap', cluster_key='leiden',
                       cell_type_key=None, fontsize=12, fontweight='bold', color='black'):
    """
    在聚类图上添加簇标签 - 无方框版本

    参数:
    - ax: 坐标轴对象
    - adata: AnnData对象
    - embedding: 嵌入类型 ('X_umap' 或 'X_tsne')
    - cluster_key: 聚类结果的列名
    - cell_type_key: 细胞类型注释的列名
    - fontsize: 字体大小
    - fontweight: 字体粗细
    - color: 文字颜色
    """
    # 计算每个簇的中心位置
    clusters = sorted(adata.obs[cluster_key].unique())

    for cluster in clusters:
        cluster_mask = adata.obs[cluster_key] == cluster
        cluster_coords = adata.obsm[embedding][cluster_mask]

        # 计算簇的中心
        center = np.median(cluster_coords, axis=0)

        # 确定标签文本
        if cell_type_key and cell_type_key in adata.obs.columns:
            # 获取该簇的细胞类型
            cell_type = adata.obs[cell_type_key][cluster_mask].iloc[0]
            label_text = f"{cluster}\n{cell_type}"
        else:
            label_text = str(cluster)

        # 添加标签 - 无方框
        ax.text(center[0], center[1], label_text,
                fontsize=fontsize, fontweight=fontweight,
                ha='center', va='center',
                color=color,
                fontname='Times New Roman')


def annotate_cell_types(adata, marker_dict):
    """
    根据标记基因自动注释细胞类型

    参数:
    - adata: AnnData对象
    - marker_dict: 标记基因字典

    返回:
    - 添加了细胞类型注释的AnnData对象
    """
    print(">>> 开始自动细胞类型注释...")

    # 创建细胞类型得分矩阵
    cell_types = list(marker_dict.keys())
    clusters = sorted(adata.obs['leiden'].cat.categories)

    # 初始化得分矩阵
    scores = pd.DataFrame(0, index=clusters, columns=cell_types)

    # 计算每个细胞类型在每个簇中的得分
    for cell_type, markers in marker_dict.items():
        # 获取在数据中存在的标记基因
        available_markers = [m for m in markers if m in adata.var_names]
        if not available_markers:
            continue

        # 计算每个簇中这些标记基因的平均表达量
        for cluster in clusters:
            cluster_mask = adata.obs['leiden'] == cluster
            cluster_data = adata[cluster_mask, available_markers]

            # 计算平均表达量
            if scipy.sparse.issparse(cluster_data.X):
                expr_values = cluster_data.X.toarray()
            else:
                expr_values = cluster_data.X

            mean_expr = np.mean(expr_values)
            scores.loc[cluster, cell_type] = mean_expr

    # 为每个簇分配细胞类型
    cell_type_annotations = {}
    for cluster in clusters:
        # 找到得分最高的细胞类型
        best_cell_type = scores.loc[cluster].idxmax()
        best_score = scores.loc[cluster].max()

        # 如果最高得分太低，标记为Unknown
        if best_score < 0.1:  # 阈值可以根据数据调整
            cell_type_annotations[cluster] = 'Unknown'
        else:
            cell_type_annotations[cluster] = best_cell_type

    # 将细胞类型注释添加到adata.obs中
    adata.obs['cell_type'] = adata.obs['leiden'].map(cell_type_annotations)

    # 打印注释结果
    print("细胞类型注释结果:")
    for cluster in clusters:
        print(f"Cluster {cluster}: {cell_type_annotations[cluster]}")

    return adata


# ================================
# 标记基因可视化函数
# ================================
def create_marker_gene_plots(adata, marker_dict, out_dir, ncols=5):
    """创建标记基因表达图 - 罗马字体版本"""
    print(">>> 创建标记基因表达图...")

    # 收集所有标记基因
    all_markers = []
    for cell_type, markers in marker_dict.items():
        all_markers.extend(markers)
    all_markers = list(set(all_markers))

    # 过滤掉数据中不存在的基因
    existing_markers = [gene for gene in all_markers if gene in adata.var_names]
    print(f"可用的标记基因: {len(existing_markers)}/{len(all_markers)}")

    if len(existing_markers) == 0:
        print("没有找到标记基因，跳过可视化")
        return

    # 按细胞类型分组显示标记基因
    for cell_type, markers in marker_dict.items():
        # 获取该细胞类型中存在的标记基因
        cell_type_markers = [gene for gene in markers if gene in adata.var_names]
        if len(cell_type_markers) == 0:
            continue

        print(f"为 {cell_type} 创建标记基因图: {len(cell_type_markers)} 个基因")

        # 计算子图布局
        nrows = math.ceil(len(cell_type_markers) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))

        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        axes = axes.flatten()

        for idx, gene in enumerate(cell_type_markers):
            if idx >= len(axes):
                break

            ax = axes[idx]
            create_clean_axis(ax)

            # 检查基因是否在数据中且表达
            if gene in adata.var_names:
                # 获取表达值
                if scipy.sparse.issparse(adata.X):
                    expr = adata[:, gene].X.toarray().flatten()
                else:
                    expr = adata[:, gene].X.flatten()

                # 创建散点图
                scatter = ax.scatter(adata.obsm['X_umap'][:, 0],
                                     adata.obsm['X_umap'][:, 1],
                                     c=expr, s=2, alpha=0.8, cmap='Reds')

                ax.set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
                ax.set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
                ax.set_title(f'{gene}', fontname='Times New Roman', fontsize=12, fontweight='bold',
                             pad=15)  # 增加标题的padding

                # 添加颜色条
                plt.colorbar(scatter, ax=ax, shrink=0.8)

            else:
                ax.text(0.5, 0.5, f'{gene}\nNot found',
                        ha='center', va='center',
                        fontname='Times New Roman', fontsize=12)
                ax.axis('off')

        # 隐藏多余的子图
        for idx in range(len(cell_type_markers), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'{cell_type} Marker Genes Expression on UMAP',
                     fontname='Times New Roman', fontsize=16, fontweight='bold', y=0.95)  # 上移标题
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局，为标题留出空间
        save_figure_multiformat(fig, f"marker_genes_{cell_type.replace(' ', '_')}_umap", out_dir)
        plt.close()

    # 创建所有标记基因的汇总图（最多显示25个基因）
    print(">>> 创建标记基因汇总图...")
    top_markers = existing_markers[:25]  # 限制数量避免图像过大
    nrows_summary = math.ceil(len(top_markers) / ncols)

    # 增加图形高度，为标题留出更多空间
    fig_height = 5 * nrows_summary + 1  # 额外增加1英寸用于标题

    fig, axes = plt.subplots(nrows_summary, ncols, figsize=(5 * ncols, fig_height))

    if nrows_summary == 1 and ncols == 1:
        axes = np.array([axes])
    elif nrows_summary == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes = axes.flatten()

    for idx, gene in enumerate(top_markers):
        if idx >= len(axes):
            break

        ax = axes[idx]
        create_clean_axis(ax)

        if gene in adata.var_names:
            # 获取表达值
            if scipy.sparse.issparse(adata.X):
                expr = adata[:, gene].X.toarray().flatten()
            else:
                expr = adata[:, gene].X.flatten()

            # 创建散点图
            scatter = ax.scatter(adata.obsm['X_umap'][:, 0],
                                 adata.obsm['X_umap'][:, 1],
                                 c=expr, s=2, alpha=0.8, cmap='Reds')

            ax.set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
            ax.set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
            ax.set_title(f'{gene}', fontname='Times New Roman', fontsize=12, fontweight='bold',
                         pad=12)  # 增加基因名称的padding

            # 添加颜色条
            plt.colorbar(scatter, ax=ax, shrink=0.7)
        else:
            ax.text(0.5, 0.5, f'{gene}\nNot found',
                    ha='center', va='center',
                    fontname='Times New Roman', fontsize=12)
            ax.axis('off')

    # 隐藏多余的子图
    for idx in range(len(top_markers), len(axes)):
        axes[idx].axis('off')

    # 将总标题移到更上方，避免重叠
    plt.suptitle('Top Marker Genes Expression on UMAP',
                 fontname='Times New Roman', fontsize=18, fontweight='bold', y=0.99)  # 上移标题到0.99位置

    # 调整布局，为标题留出更多空间
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # 顶部保留6%的空间给标题
    save_figure_multiformat(fig, "marker_genes_summary_umap", out_dir)
    plt.close()


def create_celltype_marker_dotplot(adata, marker_dict, out_dir):
    """创建细胞类型标记基因点图 - 罗马字体版本，显示cluster编号和细胞类型"""
    print(">>> 创建细胞类型标记基因点图...")

    # 准备数据
    cell_types = list(marker_dict.keys())

    # 使用已注释的细胞类型作为cluster标签
    # 获取每个leiden cluster对应的细胞类型
    cluster_celltype_map = {}
    for cluster in sorted(adata.obs['leiden'].cat.categories):
        # 获取该cluster中最常见的细胞类型
        cluster_mask = adata.obs['leiden'] == cluster
        cell_type = adata.obs['cell_type'][cluster_mask].iloc[
            0] if 'cell_type' in adata.obs.columns else f'Cluster_{cluster}'
        cluster_celltype_map[cluster] = cell_type

    # 按细胞类型分组排序clusters
    clusters_by_celltype = {}
    for cluster, cell_type in cluster_celltype_map.items():
        if cell_type not in clusters_by_celltype:
            clusters_by_celltype[cell_type] = []
        clusters_by_celltype[cell_type].append(cluster)

    # 创建显示标签：cluster编号 + 细胞类型
    display_labels = []
    original_clusters = []  # 保存原始cluster顺序

    for cell_type, cluster_list in clusters_by_celltype.items():
        for cluster in sorted(cluster_list):
            if len(cluster_list) > 1:
                # 如果同一种细胞类型有多个clusters
                display_labels.append(f"{cluster}\n({cell_type})")
            else:
                display_labels.append(f"{cluster}\n{cell_type}")
            original_clusters.append(cluster)

    clusters = original_clusters

    # 初始化得分矩阵
    mean_expr = pd.DataFrame(0, index=clusters, columns=cell_types)
    pct_cells = pd.DataFrame(0, index=clusters, columns=cell_types)

    for cell_type in cell_types:
        markers = [m for m in marker_dict[cell_type] if m in adata.var_names]
        if len(markers) == 0:
            continue

        for cluster in clusters:
            cluster_mask = adata.obs['leiden'] == cluster
            cluster_data = adata[cluster_mask, markers]

            # 平均表达
            if scipy.sparse.issparse(cluster_data.X):
                expr_values = cluster_data.X.toarray()
                mean_expr.loc[cluster, cell_type] = np.mean(expr_values)
                pct_cells.loc[cluster, cell_type] = np.mean(expr_values > 0)
            else:
                expr_values = cluster_data.X
                mean_expr.loc[cluster, cell_type] = np.mean(expr_values)
                pct_cells.loc[cluster, cell_type] = np.mean(expr_values > 0)

    # 创建点图
    fig, ax = plt.subplots(figsize=(max(12, len(cell_types) * 0.8),
                                    max(8, len(clusters) * 0.6)))
    create_clean_axis(ax)

    # 创建散点图表示
    for i, cluster in enumerate(clusters):
        for j, cell_type in enumerate(cell_types):
            if pd.isna(pct_cells.loc[cluster, cell_type]) or pct_cells.loc[cluster, cell_type] == 0:
                continue

            # 点的大小表示表达细胞比例
            size = pct_cells.loc[cluster, cell_type] * 300 + 50
            # 颜色表示平均表达量
            color_val = mean_expr.loc[cluster, cell_type]

            ax.scatter(j, i, s=size, c=[color_val], cmap='viridis',
                       vmin=0, vmax=mean_expr.max().max() if not pd.isna(mean_expr.max().max()) else 1,
                       alpha=0.8, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Marker Gene Cell Types', fontname='Times New Roman', fontsize=12)
    ax.set_ylabel('Cell Clusters (with Cell Type Annotation)', fontname='Times New Roman', fontsize=12)
    ax.set_title('Cell Type Marker Expression by Cluster',
                 fontname='Times New Roman', fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(cell_types)))
    ax.set_xticklabels(cell_types, rotation=45, ha='right',
                       fontname='Times New Roman', fontsize=10)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(display_labels, fontname='Times New Roman', fontsize=10)

    # 添加颜色条
    if not mean_expr.empty:
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(0, mean_expr.max().max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Mean Expression', fontname='Times New Roman', fontsize=12)

    plt.tight_layout()
    save_figure_multiformat(fig, "celltype_marker_dotplot", out_dir)
    plt.close()


def create_cluster_celltype_table(adata, out_dir):
    """创建cluster与细胞类型的对照表并保存"""
    print(">>> 创建cluster与细胞类型对照表...")

    # 获取每个cluster的细胞类型
    cluster_celltype_info = {}

    for cluster in sorted(adata.obs['leiden'].cat.categories):
        cluster_mask = adata.obs['leiden'] == cluster
        cell_type_counts = adata.obs['cell_type'][cluster_mask].value_counts()

        if not cell_type_counts.empty:
            main_cell_type = cell_type_counts.index[0]
            cell_count = cell_type_counts.iloc[0]
            total_cells = len(adata.obs[cluster_mask])
            percentage = (cell_count / total_cells) * 100

            cluster_celltype_info[cluster] = {
                'main_cell_type': main_cell_type,
                'cell_count': cell_count,
                'total_cells': total_cells,
                'percentage': percentage,
                'all_cell_types': ', '.join([f"{t}({c})" for t, c in cell_type_counts.items()])
            }

    # 创建DataFrame
    df = pd.DataFrame.from_dict(cluster_celltype_info, orient='index')
    df.index.name = 'Cluster'
    df = df.sort_index()

    # 保存为CSV
    csv_path = os.path.join(out_dir, "cluster_celltype_mapping.csv")
    df.to_csv(csv_path)
    print(f"cluster与细胞类型对照表已保存: {csv_path}")

    # 创建可视化表格
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    ax.axis('tight')
    ax.axis('off')

    # 准备表格数据
    table_data = []
    for cluster, info in df.iterrows():
        table_data.append([
            cluster,
            info['main_cell_type'],
            f"{info['percentage']:.1f}%",
            info['total_cells'],
            info['all_cell_types']
        ])

    # 创建表格
    table = ax.table(cellText=table_data,
                     colLabels=['Cluster', 'Main Cell Type', 'Percentage', 'Total Cells', 'All Cell Types'],
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.1, 0.2, 0.1, 0.1, 0.5])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.set_text_props(fontfamily='Times New Roman')
        if key[0] == 0:  # 标题行
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white', weight='bold', fontfamily='Times New Roman')
        else:
            if key[0] % 2 == 0:
                cell.set_facecolor('#f2f2f2')

    ax.set_title('Cluster to Cell Type Mapping',
                 fontname='Times New Roman',
                 fontsize=14,
                 fontweight='bold',
                 pad=20)

    plt.tight_layout()
    save_figure_multiformat(fig, "cluster_celltype_table", out_dir)
    plt.close()

    return df


def create_celltype_specific_plots(adata, marker_dict, out_dir):
    """为每个细胞类型创建单独的标记基因表达图 - 罗马字体版本"""
    print(">>> 为每个细胞类型创建标记基因表达图...")

    # 为每个细胞类型创建单独的图
    for cell_type, markers in marker_dict.items():
        # 获取该细胞类型中存在的标记基因
        cell_type_markers = [gene for gene in markers if gene in adata.var_names]
        if len(cell_type_markers) == 0:
            continue

        print(f"为 {cell_type} 创建详细标记基因图")

        # 创建该细胞类型的图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # 创建干净坐标轴
        for ax in axes:
            create_clean_axis(ax)

        # 子图1: UMAP显示该细胞类型
        cell_type_mask = adata.obs['cell_type'] == cell_type
        other_mask = ~cell_type_mask

        # 先绘制其他细胞
        axes[0].scatter(adata.obsm['X_umap'][other_mask, 0],
                        adata.obsm['X_umap'][other_mask, 1],
                        c='lightgray', s=1, alpha=0.3, label='Other cells')

        # 再绘制该细胞类型
        axes[0].scatter(adata.obsm['X_umap'][cell_type_mask, 0],
                        adata.obsm['X_umap'][cell_type_mask, 1],
                        c='red', s=5, alpha=0.8, label=cell_type)

        axes[0].set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
        axes[0].set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
        axes[0].set_title(f'{cell_type} Distribution on UMAP',
                          fontname='Times New Roman', fontsize=14, fontweight='bold')
        axes[0].legend(prop={'family': 'Times New Roman', 'size': 10})

        # 子图2-4: 显示前3个标记基因
        for i in range(min(3, len(cell_type_markers))):
            gene = cell_type_markers[i]
            if i + 1 < len(axes):
                ax = axes[i + 1]

                # 获取表达值
                if scipy.sparse.issparse(adata.X):
                    expr = adata[:, gene].X.toarray().flatten()
                else:
                    expr = adata[:, gene].X.flatten()

                # 创建散点图
                scatter = ax.scatter(adata.obsm['X_umap'][:, 0],
                                     adata.obsm['X_umap'][:, 1],
                                     c=expr, s=2, alpha=0.8, cmap='Reds')

                ax.set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
                ax.set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
                ax.set_title(f'{gene} Expression',
                             fontname='Times New Roman', fontsize=14, fontweight='bold')

                # 添加颜色条
                plt.colorbar(scatter, ax=ax, shrink=0.8)

        # 如果标记基因少于3个，隐藏多余的子图
        for i in range(len(cell_type_markers) + 1, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'{cell_type} - Marker Gene Analysis',
                     fontname='Times New Roman', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_figure_multiformat(fig, f"celltype_detail_{cell_type.replace(' ', '_')}", out_dir)
        plt.close()


# ================================
# 细胞通讯分析函数 - 优化版本
# ================================
def create_breast_cancer_lr_database():
    """创建乳腺癌特异的配体-受体数据库"""
    lr_pairs = {
        # 基础免疫检查点
        'PD1_PDL1': {'ligand': 'CD274', 'receptor': 'PDCD1', 'pathway': 'Immune Checkpoint'},
        'PD1_PDL2': {'ligand': 'PDCD1LG2', 'receptor': 'PDCD1', 'pathway': 'Immune Checkpoint'},
        'CTLA4_CD80': {'ligand': 'CD80', 'receptor': 'CTLA4', 'pathway': 'Immune Checkpoint'},
        'CTLA4_CD86': {'ligand': 'CD86', 'receptor': 'CTLA4', 'pathway': 'Immune Checkpoint'},
        'LAG3_FGL1': {'ligand': 'FGL1', 'receptor': 'LAG3', 'pathway': 'Immune Checkpoint'},
        'TIGIT_PVR': {'ligand': 'PVR', 'receptor': 'TIGIT', 'pathway': 'Immune Checkpoint'},
        'TIGIT_NECTIN2': {'ligand': 'NECTIN2', 'receptor': 'TIGIT', 'pathway': 'Immune Checkpoint'},
        'CD226_PVR': {'ligand': 'PVR', 'receptor': 'CD226', 'pathway': 'Immune Checkpoint'},
        'CD96_PVR': {'ligand': 'PVR', 'receptor': 'CD96', 'pathway': 'Immune Checkpoint'},

        # 基础趋化因子
        'CXCL12_CXCR4': {'ligand': 'CXCL12', 'receptor': 'CXCR4', 'pathway': 'Chemokine'},
        'CCL5_CCR5': {'ligand': 'CCL5', 'receptor': 'CCR5', 'pathway': 'Chemokine'},
        'CCL2_CCR2': {'ligand': 'CCL2', 'receptor': 'CCR2', 'pathway': 'Chemokine'},
        'CXCL12_ACKR3': {'ligand': 'CXCL12', 'receptor': 'ACKR3', 'pathway': 'Chemokine'},
        'CXCL9_CXCR3': {'ligand': 'CXCL9', 'receptor': 'CXCR3', 'pathway': 'Chemokine'},
        'CXCL10_CXCR3': {'ligand': 'CXCL10', 'receptor': 'CXCR3', 'pathway': 'Chemokine'},
        'CXCL11_CXCR3': {'ligand': 'CXCL11', 'receptor': 'CXCR3', 'pathway': 'Chemokine'},
        'CCL5_CCR1': {'ligand': 'CCL5', 'receptor': 'CCR1', 'pathway': 'Chemokine'},
        'CCL5_CCR3': {'ligand': 'CCL5', 'receptor': 'CCR3', 'pathway': 'Chemokine'},
        'CCL3_CCR1': {'ligand': 'CCL3', 'receptor': 'CCR1', 'pathway': 'Chemokine'},
        'CCL3_CCR5': {'ligand': 'CCL3', 'receptor': 'CCR5', 'pathway': 'Chemokine'},
        'CCL4_CCR5': {'ligand': 'CCL4', 'receptor': 'CCR5', 'pathway': 'Chemokine'},
        'CX3CL1_CX3CR1': {'ligand': 'CX3CL1', 'receptor': 'CX3CR1', 'pathway': 'Chemokine'},

        # 基础生长因子
        'VEGFA_FLT1': {'ligand': 'VEGFA', 'receptor': 'FLT1', 'pathway': 'Angiogenesis'},
        'VEGFA_KDR': {'ligand': 'VEGFA', 'receptor': 'KDR', 'pathway': 'Angiogenesis'},
        'EGF_EGFR': {'ligand': 'EGF', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'VEGFA_NRP1': {'ligand': 'VEGFA', 'receptor': 'NRP1', 'pathway': 'Angiogenesis'},
        'VEGFA_NRP2': {'ligand': 'VEGFA', 'receptor': 'NRP2', 'pathway': 'Angiogenesis'},
        'VEGFC_FLT4': {'ligand': 'VEGFC', 'receptor': 'FLT4', 'pathway': 'Angiogenesis'},
        'VEGFD_FLT4': {'ligand': 'VEGFD', 'receptor': 'FLT4', 'pathway': 'Angiogenesis'},
        'PGF_FLT1': {'ligand': 'PGF', 'receptor': 'FLT1', 'pathway': 'Angiogenesis'},
        'ANGPT1_TEK': {'ligand': 'ANGPT1', 'receptor': 'TEK', 'pathway': 'Angiogenesis'},
        'ANGPT2_TEK': {'ligand': 'ANGPT2', 'receptor': 'TEK', 'pathway': 'Angiogenesis'},

        # 基础细胞因子
        'TGFB1_TGFBR1': {'ligand': 'TGFB1', 'receptor': 'TGFBR1', 'pathway': 'TGF-beta'},
        'IL6_IL6R': {'ligand': 'IL6', 'receptor': 'IL6R', 'pathway': 'IL-6'},
        'IFNG_IFNGR1': {'ligand': 'IFNG', 'receptor': 'IFNGR1', 'pathway': 'Interferon'},

        # 扩展生长因子
        'TGFA_EGFR': {'ligand': 'TGFA', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'HBEGF_EGFR': {'ligand': 'HBEGF', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'EREG_EGFR': {'ligand': 'EREG', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'FGF2_FGFR1': {'ligand': 'FGF2', 'receptor': 'FGFR1', 'pathway': 'Growth Factor'},
        'FGF1_FGFR1': {'ligand': 'FGF1', 'receptor': 'FGFR1', 'pathway': 'Growth Factor'},
        'HGF_MET': {'ligand': 'HGF', 'receptor': 'MET', 'pathway': 'Growth Factor'},
        'IGF1_IGF1R': {'ligand': 'IGF1', 'receptor': 'IGF1R', 'pathway': 'Growth Factor'},
        'IGF2_IGF1R': {'ligand': 'IGF2', 'receptor': 'IGF1R', 'pathway': 'Growth Factor'},
        'IGF2_IGF2R': {'ligand': 'IGF2', 'receptor': 'IGF2R', 'pathway': 'Growth Factor'},

        # 扩展细胞因子
        'TGFB1_TGFBR2': {'ligand': 'TGFB1', 'receptor': 'TGFBR2', 'pathway': 'TGF-beta'},
        'TGFB2_TGFBR2': {'ligand': 'TGFB2', 'receptor': 'TGFBR2', 'pathway': 'TGF-beta'},
        'TGFB3_TGFBR2': {'ligand': 'TGFB3', 'receptor': 'TGFBR2', 'pathway': 'TGF-beta'},
        'IL6_IL6ST': {'ligand': 'IL6', 'receptor': 'IL6ST', 'pathway': 'IL-6'},
        'IL11_IL11RA': {'ligand': 'IL11', 'receptor': 'IL11RA', 'pathway': 'IL-6'},
        'IFNG_IFNGR2': {'ligand': 'IFNG', 'receptor': 'IFNGR2', 'pathway': 'Interferon'},
        'IL10_IL10RA': {'ligand': 'IL10', 'receptor': 'IL10RA', 'pathway': 'IL-10'},
        'IL10_IL10RB': {'ligand': 'IL10', 'receptor': 'IL10RB', 'pathway': 'IL-10'},

        # 乳腺癌特异的
        'NRG1_ERBB3': {'ligand': 'NRG1', 'receptor': 'ERBB3', 'pathway': 'HER2 Signaling'},
        'AREG_EGFR': {'ligand': 'AREG', 'receptor': 'EGFR', 'pathway': 'Estrogen Signaling'},
        # 扩展细胞粘附和基质
        'CD44_HA': {'ligand': 'HMMR', 'receptor': 'CD44', 'pathway': 'Adhesion'},
        'ITGA5_FN1': {'ligand': 'FN1', 'receptor': 'ITGA5', 'pathway': 'Adhesion'},
        'ITGAV_CDH1': {'ligand': 'CDH1', 'receptor': 'ITGAV', 'pathway': 'Adhesion'},
        'COL1A1_ITGA1': {'ligand': 'COL1A1', 'receptor': 'ITGA1', 'pathway': 'ECM'},
        'COL1A1_ITGA2': {'ligand': 'COL1A1', 'receptor': 'ITGA2', 'pathway': 'ECM'},
        'COL1A2_ITGA1': {'ligand': 'COL1A2', 'receptor': 'ITGA1', 'pathway': 'ECM'},
        'COL4A1_ITGA1': {'ligand': 'COL4A1', 'receptor': 'ITGA1', 'pathway': 'ECM'},
        'LAMB1_ITGA6': {'ligand': 'LAMB1', 'receptor': 'ITGA6', 'pathway': 'ECM'},
        'LAMC1_ITGA6': {'ligand': 'LAMC1', 'receptor': 'ITGA6', 'pathway': 'ECM'},

        # 扩展Notch信号通路
        'JAG1_NOTCH1': {'ligand': 'JAG1', 'receptor': 'NOTCH1', 'pathway': 'Notch'},
        'JAG1_NOTCH2': {'ligand': 'JAG1', 'receptor': 'NOTCH2', 'pathway': 'Notch'},
        'JAG1_NOTCH3': {'ligand': 'JAG1', 'receptor': 'NOTCH3', 'pathway': 'Notch'},
        'JAG2_NOTCH1': {'ligand': 'JAG2', 'receptor': 'NOTCH1', 'pathway': 'Notch'},
        'DLL1_NOTCH1': {'ligand': 'DLL1', 'receptor': 'NOTCH1', 'pathway': 'Notch'},
        'DLL4_NOTCH1': {'ligand': 'DLL4', 'receptor': 'NOTCH1', 'pathway': 'Notch'},

        # 扩展Wnt信号通路
        'WNT5A_FZD5': {'ligand': 'WNT5A', 'receptor': 'FZD5', 'pathway': 'Wnt'},
        'WNT5A_ROR2': {'ligand': 'WNT5A', 'receptor': 'ROR2', 'pathway': 'Wnt'},
        'WNT3A_FZD1': {'ligand': 'WNT3A', 'receptor': 'FZD1', 'pathway': 'Wnt'},
        'WNT3A_LRP6': {'ligand': 'WNT3A', 'receptor': 'LRP6', 'pathway': 'Wnt'},
        'WNT1_FZD1': {'ligand': 'WNT1', 'receptor': 'FZD1', 'pathway': 'Wnt'},

        # 其他重要通路
        'MIF_CD74': {'ligand': 'MIF', 'receptor': 'CD74', 'pathway': 'Inflammation'},
        'MIF_CXCR4': {'ligand': 'MIF', 'receptor': 'CXCR4', 'pathway': 'Inflammation'},
        'SEMA4A_PLXNB2': {'ligand': 'SEMA4A', 'receptor': 'PLXNB2', 'pathway': 'Axon Guidance'},
        'SEMA3A_NRP1': {'ligand': 'SEMA3A', 'receptor': 'NRP1', 'pathway': 'Axon Guidance'},
        'SEMA3C_NRP1': {'ligand': 'SEMA3C', 'receptor': 'NRP1', 'pathway': 'Axon Guidance'},

        # 添加新的重要通路
        'CD47_SIRPA': {'ligand': 'CD47', 'receptor': 'SIRPA', 'pathway': 'Phagocytosis'},
        'CD47_SIRPG': {'ligand': 'CD47', 'receptor': 'SIRPG', 'pathway': 'Phagocytosis'},
        'CD200_CD200R1': {'ligand': 'CD200', 'receptor': 'CD200R1', 'pathway': 'Immunomodulation'},
        'PVRL2_PVRIG': {'ligand': 'PVRL2', 'receptor': 'PVRIG', 'pathway': 'Immune Checkpoint'},
        'HHLA2_TMIGD2': {'ligand': 'HHLA2', 'receptor': 'TMIGD2', 'pathway': 'Immune Checkpoint'},
        'BTLA_HVEM': {'ligand': 'BTLA', 'receptor': 'HVEM', 'pathway': 'Immune Checkpoint'},
    }
    return lr_pairs


def create_extended_lr_database():
    """创建扩展的配体-受体数据库"""
    # CellChat数据库的精选配体-受体对
    lr_pairs = {
        # 扩展免疫检查点
        'PD1_PDL1': {'ligand': 'CD274', 'receptor': 'PDCD1', 'pathway': 'Immune Checkpoint'},
        'PD1_PDL2': {'ligand': 'PDCD1LG2', 'receptor': 'PDCD1', 'pathway': 'Immune Checkpoint'},
        'CTLA4_CD80': {'ligand': 'CD80', 'receptor': 'CTLA4', 'pathway': 'Immune Checkpoint'},
        'CTLA4_CD86': {'ligand': 'CD86', 'receptor': 'CTLA4', 'pathway': 'Immune Checkpoint'},
        'LAG3_FGL1': {'ligand': 'FGL1', 'receptor': 'LAG3', 'pathway': 'Immune Checkpoint'},
        'TIGIT_PVR': {'ligand': 'PVR', 'receptor': 'TIGIT', 'pathway': 'Immune Checkpoint'},
        'TIGIT_NECTIN2': {'ligand': 'NECTIN2', 'receptor': 'TIGIT', 'pathway': 'Immune Checkpoint'},
        'CD226_PVR': {'ligand': 'PVR', 'receptor': 'CD226', 'pathway': 'Immune Checkpoint'},
        'CD96_PVR': {'ligand': 'PVR', 'receptor': 'CD96', 'pathway': 'Immune Checkpoint'},

        # 扩展趋化因子
        'CXCL12_CXCR4': {'ligand': 'CXCL12', 'receptor': 'CXCR4', 'pathway': 'Chemokine'},
        'CXCL12_ACKR3': {'ligand': 'CXCL12', 'receptor': 'ACKR3', 'pathway': 'Chemokine'},
        'CXCL9_CXCR3': {'ligand': 'CXCL9', 'receptor': 'CXCR3', 'pathway': 'Chemokine'},
        'CXCL10_CXCR3': {'ligand': 'CXCL10', 'receptor': 'CXCR3', 'pathway': 'Chemokine'},
        'CXCL11_CXCR3': {'ligand': 'CXCL11', 'receptor': 'CXCR3', 'pathway': 'Chemokine'},
        'CCL5_CCR1': {'ligand': 'CCL5', 'receptor': 'CCR1', 'pathway': 'Chemokine'},
        'CCL5_CCR3': {'ligand': 'CCL5', 'receptor': 'CCR3', 'pathway': 'Chemokine'},
        'CCL5_CCR5': {'ligand': 'CCL5', 'receptor': 'CCR5', 'pathway': 'Chemokine'},
        'CCL2_CCR2': {'ligand': 'CCL2', 'receptor': 'CCR2', 'pathway': 'Chemokine'},
        'CCL3_CCR1': {'ligand': 'CCL3', 'receptor': 'CCR1', 'pathway': 'Chemokine'},
        'CCL3_CCR5': {'ligand': 'CCL3', 'receptor': 'CCR5', 'pathway': 'Chemokine'},
        'CCL4_CCR5': {'ligand': 'CCL4', 'receptor': 'CCR5', 'pathway': 'Chemokine'},
        'CX3CL1_CX3CR1': {'ligand': 'CX3CL1', 'receptor': 'CX3CR1', 'pathway': 'Chemokine'},

        # 扩展生长因子和血管生成
        'VEGFA_FLT1': {'ligand': 'VEGFA', 'receptor': 'FLT1', 'pathway': 'Angiogenesis'},
        'VEGFA_KDR': {'ligand': 'VEGFA', 'receptor': 'KDR', 'pathway': 'Angiogenesis'},
        'VEGFA_NRP1': {'ligand': 'VEGFA', 'receptor': 'NRP1', 'pathway': 'Angiogenesis'},
        'VEGFA_NRP2': {'ligand': 'VEGFA', 'receptor': 'NRP2', 'pathway': 'Angiogenesis'},
        'VEGFC_FLT4': {'ligand': 'VEGFC', 'receptor': 'FLT4', 'pathway': 'Angiogenesis'},
        'VEGFD_FLT4': {'ligand': 'VEGFD', 'receptor': 'FLT4', 'pathway': 'Angiogenesis'},
        'PGF_FLT1': {'ligand': 'PGF', 'receptor': 'FLT1', 'pathway': 'Angiogenesis'},
        'ANGPT1_TEK': {'ligand': 'ANGPT1', 'receptor': 'TEK', 'pathway': 'Angiogenesis'},
        'ANGPT2_TEK': {'ligand': 'ANGPT2', 'receptor': 'TEK', 'pathway': 'Angiogenesis'},

        # 扩展生长因子
        'EGF_EGFR': {'ligand': 'EGF', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'TGFA_EGFR': {'ligand': 'TGFA', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'HBEGF_EGFR': {'ligand': 'HBEGF', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'EREG_EGFR': {'ligand': 'EREG', 'receptor': 'EGFR', 'pathway': 'Growth Factor'},
        'FGF2_FGFR1': {'ligand': 'FGF2', 'receptor': 'FGFR1', 'pathway': 'Growth Factor'},
        'FGF1_FGFR1': {'ligand': 'FGF1', 'receptor': 'FGFR1', 'pathway': 'Growth Factor'},
        'HGF_MET': {'ligand': 'HGF', 'receptor': 'MET', 'pathway': 'Growth Factor'},
        'IGF1_IGF1R': {'ligand': 'IGF1', 'receptor': 'IGF1R', 'pathway': 'Growth Factor'},
        'IGF2_IGF1R': {'ligand': 'IGF2', 'receptor': 'IGF1R', 'pathway': 'Growth Factor'},
        'IGF2_IGF2R': {'ligand': 'IGF2', 'receptor': 'IGF2R', 'pathway': 'Growth Factor'},

        # 扩展细胞因子
        'TGFB1_TGFBR1': {'ligand': 'TGFB1', 'receptor': 'TGFBR1', 'pathway': 'TGF-beta'},
        'TGFB1_TGFBR2': {'ligand': 'TGFB1', 'receptor': 'TGFBR2', 'pathway': 'TGF-beta'},
        'TGFB2_TGFBR2': {'ligand': 'TGFB2', 'receptor': 'TGFBR2', 'pathway': 'TGF-beta'},
        'TGFB3_TGFBR2': {'ligand': 'TGFB3', 'receptor': 'TGFBR2', 'pathway': 'TGF-beta'},
        'IL6_IL6R': {'ligand': 'IL6', 'receptor': 'IL6R', 'pathway': 'IL-6'},
        'IL6_IL6ST': {'ligand': 'IL6', 'receptor': 'IL6ST', 'pathway': 'IL-6'},
        'IL11_IL11RA': {'ligand': 'IL11', 'receptor': 'IL11RA', 'pathway': 'IL-6'},
        'IFNG_IFNGR1': {'ligand': 'IFNG', 'receptor': 'IFNGR1', 'pathway': 'Interferon'},
        'IFNG_IFNGR2': {'ligand': 'IFNG', 'receptor': 'IFNGR2', 'pathway': 'Interferon'},
        'IL10_IL10RA': {'ligand': 'IL10', 'receptor': 'IL10RA', 'pathway': 'IL-10'},
        'IL10_IL10RB': {'ligand': 'IL10', 'receptor': 'IL10RB', 'pathway': 'IL-10'},

        # 扩展细胞粘附和基质
        'CD44_HA': {'ligand': 'HMMR', 'receptor': 'CD44', 'pathway': 'Adhesion'},
        'ITGA5_FN1': {'ligand': 'FN1', 'receptor': 'ITGA5', 'pathway': 'Adhesion'},
        'ITGAV_CDH1': {'ligand': 'CDH1', 'receptor': 'ITGAV', 'pathway': 'Adhesion'},
        'COL1A1_ITGA1': {'ligand': 'COL1A1', 'receptor': 'ITGA1', 'pathway': 'ECM'},
        'COL1A1_ITGA2': {'ligand': 'COL1A1', 'receptor': 'ITGA2', 'pathway': 'ECM'},
        'COL1A2_ITGA1': {'ligand': 'COL1A2', 'receptor': 'ITGA1', 'pathway': 'ECM'},
        'COL4A1_ITGA1': {'ligand': 'COL4A1', 'receptor': 'ITGA1', 'pathway': 'ECM'},
        'LAMB1_ITGA6': {'ligand': 'LAMB1', 'receptor': 'ITGA6', 'pathway': 'ECM'},
        'LAMC1_ITGA6': {'ligand': 'LAMC1', 'receptor': 'ITGA6', 'pathway': 'ECM'},

        # 扩展Notch信号通路
        'JAG1_NOTCH1': {'ligand': 'JAG1', 'receptor': 'NOTCH1', 'pathway': 'Notch'},
        'JAG1_NOTCH2': {'ligand': 'JAG1', 'receptor': 'NOTCH2', 'pathway': 'Notch'},
        'JAG1_NOTCH3': {'ligand': 'JAG1', 'receptor': 'NOTCH3', 'pathway': 'Notch'},
        'JAG2_NOTCH1': {'ligand': 'JAG2', 'receptor': 'NOTCH1', 'pathway': 'Notch'},
        'DLL1_NOTCH1': {'ligand': 'DLL1', 'receptor': 'NOTCH1', 'pathway': 'Notch'},
        'DLL4_NOTCH1': {'ligand': 'DLL4', 'receptor': 'NOTCH1', 'pathway': 'Notch'},

        # 扩展Wnt信号通路
        'WNT5A_FZD5': {'ligand': 'WNT5A', 'receptor': 'FZD5', 'pathway': 'Wnt'},
        'WNT5A_ROR2': {'ligand': 'WNT5A', 'receptor': 'ROR2', 'pathway': 'Wnt'},
        'WNT3A_FZD1': {'ligand': 'WNT3A', 'receptor': 'FZD1', 'pathway': 'Wnt'},
        'WNT3A_LRP6': {'ligand': 'WNT3A', 'receptor': 'LRP6', 'pathway': 'Wnt'},
        'WNT1_FZD1': {'ligand': 'WNT1', 'receptor': 'FZD1', 'pathway': 'Wnt'},

        # 其他重要通路
        'MIF_CD74': {'ligand': 'MIF', 'receptor': 'CD74', 'pathway': 'Inflammation'},
        'MIF_CXCR4': {'ligand': 'MIF', 'receptor': 'CXCR4', 'pathway': 'Inflammation'},
        'SEMA4A_PLXNB2': {'ligand': 'SEMA4A', 'receptor': 'PLXNB2', 'pathway': 'Axon Guidance'},
        'SEMA3A_NRP1': {'ligand': 'SEMA3A', 'receptor': 'NRP1', 'pathway': 'Axon Guidance'},
        'SEMA3C_NRP1': {'ligand': 'SEMA3C', 'receptor': 'NRP1', 'pathway': 'Axon Guidance'},

        # 添加新的重要通路
        'CD47_SIRPA': {'ligand': 'CD47', 'receptor': 'SIRPA', 'pathway': 'Phagocytosis'},
        'CD47_SIRPG': {'ligand': 'CD47', 'receptor': 'SIRPG', 'pathway': 'Phagocytosis'},
        'CD200_CD200R1': {'ligand': 'CD200', 'receptor': 'CD200R1', 'pathway': 'Immunomodulation'},
        'PVRL2_PVRIG': {'ligand': 'PVRL2', 'receptor': 'PVRIG', 'pathway': 'Immune Checkpoint'},
        'HHLA2_TMIGD2': {'ligand': 'HHLA2', 'receptor': 'TMIGD2', 'pathway': 'Immune Checkpoint'},
        'BTLA_HVEM': {'ligand': 'BTLA', 'receptor': 'HVEM', 'pathway': 'Immune Checkpoint'},
    }
    return lr_pairs


def validate_lr_database(adata, lr_database):
    """验证配体-受体数据库中的基因是否在数据中存在"""
    available_pairs = {}
    missing_genes = {}

    for lr_name, lr_info in lr_database.items():
        ligand = lr_info['ligand']
        receptor = lr_info['receptor']

        ligand_exists = ligand in adata.var_names
        receptor_exists = receptor in adata.var_names

        if ligand_exists and receptor_exists:
            available_pairs[lr_name] = lr_info
        else:
            missing_genes[lr_name] = {
                'ligand': ligand if not ligand_exists else None,
                'receptor': receptor if not receptor_exists else None
            }

    print(f"可用的配体-受体对: {len(available_pairs)}/{len(lr_database)}")
    print(f"缺失基因的对数: {len(missing_genes)}")

    if missing_genes:
        print("\n缺失基因的配体-受体对:")
        for pair, missing in list(missing_genes.items())[:10]:  # 只显示前10个
            print(f"  {pair}: {missing}")

    return available_pairs


def analyze_ligand_receptor_expression(adata, lr_database, cluster_key):
    """分析配体-受体表达"""
    results = []
    clusters = sorted(adata.obs[cluster_key].unique())

    print(f"分析 {len(lr_database)} 个配体-受体对在 {len(clusters)} 个细胞群中的表达...")

    for lr_name, lr_info in lr_database.items():
        ligand = lr_info['ligand']
        receptor = lr_info['receptor']
        pathway = lr_info['pathway']

        # 检查基因是否在数据中
        if ligand in adata.var_names and receptor in adata.var_names:
            for source_cluster in clusters:
                for target_cluster in clusters:
                    # 计算配体在source cluster的表达
                    source_mask = adata.obs[cluster_key] == source_cluster
                    if scipy.sparse.issparse(adata.X):
                        ligand_expr = np.mean(adata[source_mask, ligand].X.toarray())
                    else:
                        ligand_expr = np.mean(adata[source_mask, ligand].X)

                    # 计算受体在target cluster的表达
                    target_mask = adata.obs[cluster_key] == target_cluster
                    if scipy.sparse.issparse(adata.X):
                        receptor_expr = np.mean(adata[target_mask, receptor].X.toarray())
                    else:
                        receptor_expr = np.mean(adata[target_mask, receptor].X)

                    # 计算通讯分数（配体表达 × 受体表达）
                    communication_score = ligand_expr * receptor_expr

                    # 计算表达细胞比例
                    if scipy.sparse.issparse(adata.X):
                        ligand_cell_ratio = np.mean(adata[source_mask, ligand].X.toarray() > 0)
                        receptor_cell_ratio = np.mean(adata[target_mask, receptor].X.toarray() > 0)
                    else:
                        ligand_cell_ratio = np.mean(adata[source_mask, ligand].X > 0)
                        receptor_cell_ratio = np.mean(adata[target_mask, receptor].X > 0)

                    results.append({
                        'ligand_receptor_pair': lr_name,
                        'ligand': ligand,
                        'receptor': receptor,
                        'pathway': pathway,
                        'source_cluster': source_cluster,
                        'target_cluster': target_cluster,
                        'ligand_expression': ligand_expr,
                        'receptor_expression': receptor_expr,
                        'communication_score': communication_score,
                        'ligand_cell_ratio': ligand_cell_ratio,
                        'receptor_cell_ratio': receptor_cell_ratio
                    })
        else:
            # 记录缺失的基因
            missing_genes = []
            if ligand not in adata.var_names:
                missing_genes.append(ligand)
            if receptor not in adata.var_names:
                missing_genes.append(receptor)
            print(f"跳过 {lr_name}: 缺少基因 {missing_genes}")

    return pd.DataFrame(results)


def add_communication_metrics(results_df, adata, cluster_key):
    """添加额外的通讯指标"""
    # 计算标准化通讯分数
    max_score = results_df['communication_score'].max()
    if max_score > 0:
        results_df['normalized_score'] = results_df['communication_score'] / max_score
    else:
        results_df['normalized_score'] = 0

    # 计算显著性分数（基于表达细胞比例）
    results_df['significance_score'] = (
            results_df['ligand_cell_ratio'] * results_df['receptor_cell_ratio'] *
            results_df['normalized_score']
    )

    # 添加相互作用类型
    def get_interaction_type(row):
        if row['source_cluster'] == row['target_cluster']:
            return 'autocrine'
        else:
            return 'paracrine'

    results_df['interaction_type'] = results_df.apply(get_interaction_type, axis=1)

    return results_df


def get_cluster_labels(adata, cluster_key='leiden', cell_type_key='cell_type'):
    """获取cluster的标签（包含细胞类型名称）"""
    cluster_labels = {}
    for cluster in sorted(adata.obs[cluster_key].unique()):
        cluster_mask = adata.obs[cluster_key] == cluster
        # 获取主要细胞类型
        if cell_type_key in adata.obs.columns:
            cell_type = adata.obs[cell_type_key][cluster_mask].iloc[0]
            # 如果细胞类型太长，进行简化
            if len(cell_type) > 20:
                # 提取主要部分
                if 'cells' in cell_type.lower():
                    parts = cell_type.split('_')
                    if len(parts) > 1:
                        cell_type = parts[-1]
                    else:
                        cell_type = cell_type[:20] + '...'
            # 创建标签：cluster编号 + 细胞类型（换行显示）
            cluster_labels[cluster] = f"{cluster}\n{cell_type}"
        else:
            cluster_labels[cluster] = str(cluster)

    return cluster_labels


def create_communication_heatmap_with_cell_types(results_df, out_dir, adata, cluster_key='leiden'):
    """创建通讯热图 - 使用细胞类型名称而非数字"""
    try:
        # 获取cluster标签
        cluster_labels = get_cluster_labels(adata, cluster_key)

        # 创建通讯矩阵
        clusters = sorted(set(results_df['source_cluster']).union(set(results_df['target_cluster'])))

        # 使用原始cluster顺序，但标签使用细胞类型名称
        display_labels = [cluster_labels.get(str(cluster), str(cluster)) for cluster in clusters]

        comm_matrix = pd.DataFrame(0, index=clusters, columns=clusters)

        for _, row in results_df.iterrows():
            comm_matrix.loc[row['source_cluster'], row['target_cluster']] += row['communication_score']

        fig, ax = plt.subplots(figsize=(max(10, len(clusters) * 0.5),
                                        max(8, len(clusters) * 0.5)))
        create_clean_axis(ax)

        im = ax.imshow(comm_matrix.values, cmap='viridis', aspect='auto')

        # 设置刻度标签
        ax.set_xticks(range(len(clusters)))
        ax.set_yticks(range(len(clusters)))

        # 使用细胞类型名称作为标签
        ax.set_xticklabels(display_labels,
                           fontname='Times New Roman',
                           fontsize=9,
                           rotation=45,
                           ha='right')
        ax.set_yticklabels(display_labels,
                           fontname='Times New Roman',
                           fontsize=9)

        ax.set_xlabel('Target Cell Type', fontname='Times New Roman', fontsize=12)
        ax.set_ylabel('Source Cell Type', fontname='Times New Roman', fontsize=12)
        ax.set_title('Cell-Cell Communication Network by Cell Type',
                     fontname='Times New Roman',
                     fontsize=16,
                     fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Communication Score', fontname='Times New Roman', fontsize=11)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(9)

        plt.tight_layout()
        save_figure_multiformat(fig, "communication_network_by_celltype", out_dir)
        plt.close()

        print("基于细胞类型的通讯网络热图已生成")

    except Exception as e:
        print(f"创建基于细胞类型的通讯热图时出错: {e}")


def create_ligand_receptor_bubble_with_names(results_df, out_dir, adata, cluster_key='leiden'):
    """创建配体-受体气泡图 - 使用细胞类型名称"""
    try:
        # 获取cluster标签
        cluster_labels = get_cluster_labels(adata, cluster_key)

        # 选择通讯分数最高的配体-受体对
        top_pairs = results_df.groupby('ligand_receptor_pair')['communication_score'].max().nlargest(20)

        if len(top_pairs) == 0:
            print("没有找到显著的配体-受体对")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        create_clean_axis(ax)

        # 创建气泡图数据
        bubble_data = []
        for pair in top_pairs.index:
            pair_data = results_df[results_df['ligand_receptor_pair'] == pair]

            # 找到通讯最强的source-target组合
            max_idx = pair_data['communication_score'].idxmax()
            max_row = pair_data.loc[max_idx]

            # 使用细胞类型名称
            source_label = cluster_labels.get(str(max_row['source_cluster']), str(max_row['source_cluster']))
            target_label = cluster_labels.get(str(max_row['target_cluster']), str(max_row['target_cluster']))

            bubble_data.append({
                'pair': pair,
                'pathway': max_row['pathway'],
                'score': max_row['communication_score'],
                'ligand': max_row['ligand'],
                'receptor': max_row['receptor'],
                'source': source_label,
                'target': target_label,
                'interaction': f"{source_label} → {target_label}"
            })

        bubble_df = pd.DataFrame(bubble_data)

        # 按通路分组
        pathways = bubble_df['pathway'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(pathways)))
        color_map = {pathway: colors[i] for i, pathway in enumerate(pathways)}

        # 创建气泡图
        for i, row in bubble_df.iterrows():
            ax.scatter(row['score'], i,
                       s=row['score'] * 5000,
                       c=[color_map[row['pathway']]],
                       alpha=0.7, edgecolors='black', linewidth=0.5)

            # 添加交互信息
            ax.text(row['score'] + row['score'] * 0.05, i,
                    f"{row['interaction']}",
                    fontname='Times New Roman', fontsize=8,
                    va='center')

        ax.set_xlabel('Max Communication Score', fontname='Times New Roman', fontsize=12)
        ax.set_ylabel('Ligand-Receptor Pairs', fontname='Times New Roman', fontsize=12)
        ax.set_title('Top Ligand-Receptor Interactions by Cell Type',
                     fontname='Times New Roman',
                     fontsize=16,
                     fontweight='bold')

        # 设置Y轴标签
        y_labels = []
        for _, row in bubble_df.iterrows():
            y_labels.append(f"{row['ligand']}-{row['receptor']}\n({row['pair']})")

        ax.set_yticks(range(len(bubble_df)))
        ax.set_yticklabels(y_labels, fontname='Times New Roman', fontsize=9)

        # 创建图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=color_map[pathway], markersize=8, label=pathway)
                           for pathway in pathways]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
                  loc='upper left', prop={'family': 'Times New Roman', 'size': 9})

        plt.tight_layout()
        save_figure_multiformat(fig, "ligand_receptor_bubble_with_names", out_dir)
        plt.close()

        print("基于细胞类型的配体-受体气泡图已生成")

    except Exception as e:
        print(f"创建基于细胞类型的配体-受体气泡图时出错: {e}")


def create_pathway_heatmap_with_cell_types(results_df, pathway_name, out_dir, adata, cluster_key='leiden'):
    """创建特定通路的通讯热图 - 使用细胞类型名称"""
    try:
        # 获取cluster标签
        cluster_labels = get_cluster_labels(adata, cluster_key)

        # 筛选该通路的数据
        pathway_data = results_df[results_df['pathway'] == pathway_name]

        if len(pathway_data) == 0:
            print(f"未找到通路 {pathway_name} 的数据")
            return

        # 创建通讯矩阵
        clusters = sorted(set(pathway_data['source_cluster']).union(set(pathway_data['target_cluster'])))

        # 使用细胞类型名称作为标签
        display_labels = [cluster_labels.get(str(cluster), str(cluster)) for cluster in clusters]

        pathway_matrix = pd.DataFrame(0, index=clusters, columns=clusters)

        for _, row in pathway_data.iterrows():
            pathway_matrix.loc[row['source_cluster'], row['target_cluster']] += row['communication_score']

        fig, ax = plt.subplots(figsize=(max(8, len(clusters) * 0.6),
                                        max(6, len(clusters) * 0.6)))
        create_clean_axis(ax)

        im = ax.imshow(pathway_matrix.values, cmap='Reds', aspect='auto')

        # 设置刻度标签
        ax.set_xticks(range(len(clusters)))
        ax.set_yticks(range(len(clusters)))
        ax.set_xticklabels(display_labels,
                           fontname='Times New Roman',
                           fontsize=8,
                           rotation=45,
                           ha='right')
        ax.set_yticklabels(display_labels,
                           fontname='Times New Roman',
                           fontsize=8)

        ax.set_xlabel('Target Cell Type', fontname='Times New Roman', fontsize=11)
        ax.set_ylabel('Source Cell Type', fontname='Times New Roman', fontsize=11)
        ax.set_title(f'{pathway_name} Pathway Communication by Cell Type',
                     fontname='Times New Roman', fontsize=14, fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Communication Score', fontname='Times New Roman', fontsize=10)
        for label in cbar.ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(8)

        plt.tight_layout()
        save_figure_multiformat(fig, f"pathway_{pathway_name.replace(' ', '_')}_by_celltype", out_dir)
        plt.close()

        print(f"通路 {pathway_name} 的基于细胞类型热图已生成")

    except Exception as e:
        print(f"创建通路 {pathway_name} 的热图时出错: {e}")


def create_pathway_analysis_with_names(results_df, out_dir, adata, cluster_key='leiden'):
    """创建信号通路分析 - 使用细胞类型名称"""
    try:
        # 获取cluster标签
        cluster_labels = get_cluster_labels(adata, cluster_key)

        # 按通路汇总通讯分数
        pathway_summary = results_df.groupby('pathway').agg({
            'communication_score': ['sum', 'mean', 'max', 'count']
        }).round(4)
        pathway_summary.columns = ['_'.join(col).strip() for col in pathway_summary.columns.values]
        pathway_summary = pathway_summary.sort_values('communication_score_sum', ascending=False)

        # 绘制通路活性条形图
        fig, ax = plt.subplots(figsize=(12, 8))
        create_clean_axis(ax)

        pathways = pathway_summary.index
        y_pos = np.arange(len(pathways))

        bars = ax.barh(y_pos, pathway_summary['communication_score_sum'],
                       color='skyblue', edgecolor='black', alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pathways, fontname='Times New Roman', fontsize=10)
        ax.set_xlabel('Total Communication Score', fontname='Times New Roman', fontsize=12)
        ax.set_title('Pathway Activity in Cell-Cell Communication',
                     fontname='Times New Roman', fontsize=16, fontweight='bold')

        # 在条形上添加数值
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:.2f}', ha='left', va='center', fontname='Times New Roman')

        plt.tight_layout()
        save_figure_multiformat(fig, "pathway_activity_by_name", out_dir)
        plt.close()

        print("基于细胞类型的信号通路分析图已生成")

        # 为重要通路创建详细热图
        important_pathways = ['Angiogenesis', 'Chemokine', 'ECM', 'Notch', 'Wnt', 'TGF-beta', 'Immune Checkpoint']

        for pathway in important_pathways:
            if pathway in results_df['pathway'].unique():
                create_pathway_heatmap_with_cell_types(results_df, pathway, out_dir, adata, cluster_key)

    except Exception as e:
        print(f"创建基于细胞类型的信号通路分析图时出错: {e}")


def create_cell_role_analysis_with_names(results_df, out_dir, adata, cluster_key='leiden'):
    """创建细胞角色分析 - 使用细胞类型名称"""
    try:
        # 获取cluster标签
        cluster_labels = get_cluster_labels(adata, cluster_key)

        # 计算每个细胞群的发送和接收能力
        clusters = sorted(set(results_df['source_cluster']).union(set(results_df['target_cluster'])))

        sending_strength = results_df.groupby('source_cluster')['communication_score'].sum()
        receiving_strength = results_df.groupby('target_cluster')['communication_score'].sum()

        # 确保所有cluster都在结果中
        for cluster in clusters:
            if cluster not in sending_strength:
                sending_strength[cluster] = 0
            if cluster not in receiving_strength:
                receiving_strength[cluster] = 0

        # 获取显示标签
        display_labels = [cluster_labels.get(str(cluster), str(cluster)) for cluster in clusters]

        # 按发送能力排序
        sending_sorted = sorted([(cluster, sending_strength[cluster]) for cluster in clusters],
                                key=lambda x: x[1], reverse=True)
        sending_clusters = [cluster for cluster, _ in sending_sorted]
        sending_values = [value for _, value in sending_sorted]
        sending_labels = [cluster_labels.get(str(cluster), str(cluster)) for cluster in sending_clusters]

        # 按接收能力排序
        receiving_sorted = sorted([(cluster, receiving_strength[cluster]) for cluster in clusters],
                                  key=lambda x: x[1], reverse=True)
        receiving_clusters = [cluster for cluster, _ in receiving_sorted]
        receiving_values = [value for _, value in receiving_sorted]
        receiving_labels = [cluster_labels.get(str(cluster), str(cluster)) for cluster in receiving_clusters]

        # 绘制发送和接收能力图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        create_clean_axis(ax1)
        create_clean_axis(ax2)

        # 发送能力
        bars1 = ax1.bar(range(len(sending_sorted)), sending_values,
                        color='lightcoral', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Cell Type', fontname='Times New Roman', fontsize=12)
        ax1.set_ylabel('Outgoing Communication Strength', fontname='Times New Roman', fontsize=12)
        ax1.set_title('Signal Sending Strength\n(Outgoing Communication)',
                      fontname='Times New Roman', fontsize=16, fontweight='bold')
        ax1.set_xticks(range(len(sending_sorted)))
        ax1.set_xticklabels(sending_labels, rotation=45, ha='right',
                            fontname='Times New Roman', fontsize=9)

        # 接收能力
        bars2 = ax2.bar(range(len(receiving_sorted)), receiving_values,
                        color='lightblue', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Cell Type', fontname='Times New Roman', fontsize=12)
        ax2.set_ylabel('Incoming Communication Strength', fontname='Times New Roman', fontsize=12)
        ax2.set_title('Signal Receiving Strength\n(Incoming Communication)',
                      fontname='Times New Roman', fontsize=16, fontweight='bold')
        ax2.set_xticks(range(len(receiving_sorted)))
        ax2.set_xticklabels(receiving_labels, rotation=45, ha='right',
                            fontname='Times New Roman', fontsize=9)

        plt.tight_layout()
        save_figure_multiformat(fig, "cell_communication_roles_by_name", out_dir)
        plt.close()

        print("基于细胞类型的细胞角色分析图已生成")

        # 保存详细的发送和接收能力数据
        role_data = []
        for cluster in clusters:
            role_data.append({
                'cluster': cluster,
                'cell_type_label': cluster_labels.get(str(cluster), str(cluster)),
                'sending_strength': sending_strength.get(cluster, 0),
                'receiving_strength': receiving_strength.get(cluster, 0),
                'total_communication': sending_strength.get(cluster, 0) + receiving_strength.get(cluster, 0)
            })

        role_df = pd.DataFrame(role_data)
        role_df = role_df.sort_values('total_communication', ascending=False)
        role_df.to_csv(os.path.join(out_dir, "detailed_cell_communication_roles_by_name.csv"), index=False)

    except Exception as e:
        print(f"创建基于细胞类型的细胞角色分析图时出错: {e}")


def create_interaction_type_analysis_with_names(results_df, out_dir, adata, cluster_key='leiden'):
    """创建相互作用类型分析 - 使用细胞类型名称"""
    try:
        # 获取cluster标签
        cluster_labels = get_cluster_labels(adata, cluster_key)

        # 分析自分泌vs旁分泌通讯
        interaction_summary = results_df.groupby('interaction_type').agg({
            'communication_score': ['sum', 'mean', 'count'],
            'ligand_receptor_pair': 'nunique'
        }).round(4)

        # 绘制相互作用类型分布
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        create_clean_axis(axes[0])
        create_clean_axis(axes[1])

        # 通讯分数分布
        interaction_types = results_df['interaction_type'].unique()
        scores_by_type = [results_df[results_df['interaction_type'] == t]['communication_score'] for t in
                          interaction_types]

        axes[0].boxplot(scores_by_type, labels=interaction_types)
        axes[0].set_xlabel('Interaction Type', fontname='Times New Roman', fontsize=11)
        axes[0].set_ylabel('Communication Score', fontname='Times New Roman', fontsize=11)
        axes[0].set_title('Communication Score by Interaction Type',
                          fontname='Times New Roman', fontsize=14, fontweight='bold')

        # 设置坐标轴标签字体
        for label in axes[0].get_xticklabels():
            label.set_fontname('Times New Roman')
        for label in axes[0].get_yticklabels():
            label.set_fontname('Times New Roman')

        # 相互作用类型计数
        type_counts = results_df['interaction_type'].value_counts()
        bars = axes[1].bar(type_counts.index, type_counts.values, color=['lightcoral', 'lightblue'], alpha=0.7)
        axes[1].set_xlabel('Interaction Type', fontname='Times New Roman', fontsize=11)
        axes[1].set_ylabel('Number of Interactions', fontname='Times New Roman', fontsize=11)
        axes[1].set_title('Interaction Type Distribution',
                          fontname='Times New Roman', fontsize=14, fontweight='bold')

        # 设置坐标轴标签字体
        for label in axes[1].get_xticklabels():
            label.set_fontname('Times New Roman')
        for label in axes[1].get_yticklabels():
            label.set_fontname('Times New Roman')

        # 在柱状图上添加数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height + 1,
                         f'{int(height)}', ha='center', va='bottom', fontname='Times New Roman')

        plt.tight_layout()
        save_figure_multiformat(fig, "interaction_type_analysis_by_name", out_dir)
        plt.close()

        print("基于细胞类型的相互作用类型分析图已生成")

        # 创建自分泌和旁分泌通讯的详细表格
        autocrine_data = results_df[results_df['interaction_type'] == 'autocrine']
        paracrine_data = results_df[results_df['interaction_type'] == 'paracrine']

        # 为自分泌通讯添加细胞类型信息
        if len(autocrine_data) > 0:
            autocrine_data = autocrine_data.copy()
            autocrine_data['cell_type'] = autocrine_data['source_cluster'].apply(
                lambda x: cluster_labels.get(str(x), str(x)))
            autocrine_data.to_csv(os.path.join(out_dir, "autocrine_communication_by_celltype.csv"), index=False)

        # 为旁分泌通讯添加细胞类型信息
        if len(paracrine_data) > 0:
            paracrine_data = paracrine_data.copy()
            paracrine_data['source_cell_type'] = paracrine_data['source_cluster'].apply(
                lambda x: cluster_labels.get(str(x), str(x)))
            paracrine_data['target_cell_type'] = paracrine_data['target_cluster'].apply(
                lambda x: cluster_labels.get(str(x), str(x)))
            paracrine_data.to_csv(os.path.join(out_dir, "paracrine_communication_by_celltype.csv"), index=False)

    except Exception as e:
        print(f"创建基于细胞类型的相互作用类型分析图时出错: {e}")


def create_enhanced_communication_visualizations_with_names(results_df, out_dir, adata, cluster_key='leiden'):
    """创建增强的通讯分析可视化 - 使用细胞类型名称"""
    try:
        print(">>> 创建基于细胞类型的通讯分析可视化...")

        # 1. 基于细胞类型的通讯网络热图
        create_communication_heatmap_with_cell_types(results_df, out_dir, adata, cluster_key)

        # 2. 基于细胞类型的配体-受体对气泡图
        create_ligand_receptor_bubble_with_names(results_df, out_dir, adata, cluster_key)

        # 3. 基于细胞类型的信号通路分析
        create_pathway_analysis_with_names(results_df, out_dir, adata, cluster_key)

        # 4. 基于细胞类型的细胞角色分析
        create_cell_role_analysis_with_names(results_df, out_dir, adata, cluster_key)

        # 5. 基于细胞类型的相互作用类型分析
        create_interaction_type_analysis_with_names(results_df, out_dir, adata, cluster_key)

        print(">>> 基于细胞类型的通讯可视化已生成")

    except Exception as e:
        print(f"创建基于细胞类型的通讯可视化时出错: {e}")


def save_enhanced_communication_results_with_names(results_df, out_dir, lr_database, adata, cluster_key='leiden'):
    """保存增强的通讯分析结果 - 包含细胞类型名称"""
    try:
        # 获取cluster标签
        cluster_labels = get_cluster_labels(adata, cluster_key)

        # 创建包含细胞类型名称的结果副本
        results_with_names = results_df.copy()

        # 添加细胞类型名称列
        results_with_names['source_cell_type'] = results_with_names['source_cluster'].apply(
            lambda x: cluster_labels.get(str(x), str(x)))
        results_with_names['target_cell_type'] = results_with_names['target_cluster'].apply(
            lambda x: cluster_labels.get(str(x), str(x)))

        # 保存完整结果（包含细胞类型名称）
        results_with_names.to_csv(os.path.join(out_dir, "communication_analysis_full_with_celltypes.csv"), index=False)

        # 保存汇总结果
        summary = results_with_names.groupby(['ligand_receptor_pair', 'pathway']).agg({
            'communication_score': ['max', 'mean', 'count'],
            'ligand_expression': 'mean',
            'receptor_expression': 'mean',
            'ligand_cell_ratio': 'mean',
            'receptor_cell_ratio': 'mean',
            'normalized_score': 'mean',
            'significance_score': 'mean'
        }).round(4)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary.reset_index().to_csv(os.path.join(out_dir, "communication_summary_with_celltypes.csv"), index=False)

        # 保存通路汇总（包含细胞类型信息）
        pathway_summary = results_with_names.groupby('pathway').agg({
            'communication_score': ['sum', 'mean', 'max', 'count'],
            'ligand_receptor_pair': 'nunique',
            'normalized_score': 'sum',
            'significance_score': 'sum'
        }).round(4)
        pathway_summary.columns = ['_'.join(col).strip() for col in pathway_summary.columns.values]
        pathway_summary.reset_index().to_csv(os.path.join(out_dir, "pathway_summary_with_celltypes.csv"), index=False)

        # 保存细胞角色（包含细胞类型名称）
        sending_strength = results_with_names.groupby('source_cell_type')['communication_score'].sum()
        receiving_strength = results_with_names.groupby('target_cell_type')['communication_score'].sum()

        cell_roles = pd.DataFrame({
            'sending_strength': sending_strength,
            'receiving_strength': receiving_strength
        }).reset_index()
        cell_roles.columns = ['cell_type', 'sending_strength', 'receiving_strength']
        cell_roles.to_csv(os.path.join(out_dir, "cell_communication_roles_with_celltypes.csv"), index=False)

        print(">>> 基于细胞类型的通讯分析结果已保存")

    except Exception as e:
        print(f"保存基于细胞类型的通讯分析结果时出错: {e}")


def run_optimized_communication_analysis_with_names(adata, out_dir, cluster_key='leiden'):
    """
    优化的细胞通讯分析 - 使用细胞类型名称而非数字
    """
    print("\n>>> 开始优化的细胞通讯分析（使用细胞类型名称）...")

    comm_dir = os.path.join(out_dir, "communication_analysis_with_names")
    os.makedirs(comm_dir, exist_ok=True)

    try:
        # 使用扩展的数据库
        lr_database = create_extended_lr_database()

        # 验证数据库
        available_pairs = validate_lr_database(adata, lr_database)

        if len(available_pairs) == 0:
            print("错误: 没有可用的配体-受体对!")
            return False

        # 分析配体-受体表达
        results = analyze_ligand_receptor_expression(adata, available_pairs, cluster_key)

        # 添加额外的分析
        results = add_communication_metrics(results, adata, cluster_key)

        # 创建可视化 - 使用细胞类型名称版本
        create_enhanced_communication_visualizations_with_names(results, comm_dir, adata, cluster_key)

        # 保存结果 - 包含细胞类型名称
        save_enhanced_communication_results_with_names(results, comm_dir, available_pairs, adata, cluster_key)

        print("优化的细胞通讯分析（使用细胞类型名称）完成!")
        return True

    except Exception as e:
        print(f"细胞通讯分析出错: {e}")
        import traceback
        traceback.print_exc()
        return False


# ================================
# Step 1: 读取数据
# ================================
print("\n>>> 正在读取矩阵和元信息...")

for file_path in [matrix_file, barcodes_file, genes_file]:
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        exit()

try:
    adata = sc.read_mtx(matrix_file).T
    print(f"原始数据形状: {adata.shape}")

    barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")[0].astype(str).tolist()
    adata.obs_names = barcodes

    genes_df = pd.read_csv(genes_file, header=None, sep="\t")
    print(f"基因文件形状: {genes_df.shape}")

    if genes_df.shape[1] == 1:
        genes = genes_df[0].astype(str).tolist()
        adata.var_names = genes
        adata.var['gene_ids'] = genes
    else:
        genes = genes_df[0].astype(str).tolist()
        adata.var_names = genes
        if genes_df.shape[1] >= 2:
            adata.var['gene_ids'] = genes_df[1].astype(str).tolist()

    adata.var_names_make_unique()

    if os.path.exists(metadata_file):
        metadata = pd.read_csv(metadata_file, index_col=0)
        metadata.index = metadata.index.astype(str)
        common_cells = adata.obs_names.intersection(metadata.index)
        if len(common_cells) > 0:
            adata = adata[common_cells, :]
            adata.obs = adata.obs.join(metadata)
            print(f"成功合并metadata，共同细胞数: {len(common_cells)}")
        else:
            print("警告: metadata中没有匹配的细胞")
    else:
        print("未找到metadata.csv文件，跳过元数据合并")

    print(f"最终数据: 细胞数: {adata.n_obs}, 基因数: {adata.n_vars}")

    adata = optimize_memory_usage(adata)

except Exception as e:
    print(f"读取数据时出错: {e}")
    exit()

# ================================
# Step 2: 质量控制
# ================================
print("\n>>> 进行质量控制...")

adata.var["mt"] = adata.var_names.str.startswith(("MT-", "mt-", "MT.", "mt."))
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL", "rps", "rpl"))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"],
                           percent_top=None, log1p=False, inplace=True)

qc_before = {
    'cells': adata.n_obs,
    'genes': adata.n_vars,
    'median_genes': np.median(adata.obs['n_genes_by_counts']),
    'median_umi': np.median(adata.obs['total_counts']),
    'median_mt': np.median(adata.obs['pct_counts_mt'])
}


def create_qc_plots(adata, out_dir):
    """创建质量控制图表 - 罗马字体无网格版本"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # 应用干净样式到所有坐标轴
    for ax in axes:
        create_clean_axis(ax)

    # 基因数分布
    axes[0].hist(adata.obs['n_genes_by_counts'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(config.MIN_GENES, color='red', linestyle='--', linewidth=2,
                    label=f'Min {config.MIN_GENES} genes')
    axes[0].set_xlabel('Genes per cell', fontname='Times New Roman', fontsize=12)
    axes[0].set_ylabel('Number of cells', fontname='Times New Roman', fontsize=12)
    axes[0].legend(prop={'family': 'Times New Roman'})
    axes[0].set_title('Distribution of Genes per Cell', fontname='Times New Roman', fontsize=16, fontweight='bold')

    # UMI计数分布
    axes[1].hist(adata.obs['total_counts'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('UMI counts per cell', fontname='Times New Roman', fontsize=12)
    axes[1].set_ylabel('Number of cells', fontname='Times New Roman', fontsize=12)
    axes[1].set_title('Distribution of UMI Counts', fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 线粒体基因百分比
    axes[2].hist(adata.obs['pct_counts_mt'], bins=50, alpha=0.7, color='salmon', edgecolor='black')
    axes[2].axvline(config.MAX_MT_PERCENT, color='red', linestyle='--', linewidth=2,
                    label=f'MT% < {config.MAX_MT_PERCENT}')
    axes[2].set_xlabel('Mitochondrial %', fontname='Times New Roman', fontsize=12)
    axes[2].set_ylabel('Number of cells', fontname='Times New Roman', fontsize=12)
    axes[2].legend(prop={'family': 'Times New Roman'})
    axes[2].set_title('Distribution of Mitochondrial %', fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 散点图
    axes[3].scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'], alpha=0.6, s=24)
    axes[3].set_xlabel('UMI counts', fontname='Times New Roman', fontsize=12)
    axes[3].set_ylabel('Genes per cell', fontname='Times New Roman', fontsize=12)
    axes[3].set_title('UMI vs Genes', fontname='Times New Roman', fontsize=16, fontweight='bold')

    axes[4].scatter(adata.obs['total_counts'], adata.obs['pct_counts_mt'], alpha=0.6, s=10, color='orange')
    axes[4].set_xlabel('UMI counts', fontname='Times New Roman', fontsize=12)
    axes[4].set_ylabel('Mitochondrial %', fontname='Times New Roman', fontsize=12)
    axes[4].set_title('UMI vs MT%', fontname='Times New Roman', fontsize=16, fontweight='bold')

    axes[5].scatter(adata.obs['n_genes_by_counts'], adata.obs['pct_counts_mt'], alpha=0.6, s=10, color='green')
    axes[5].set_xlabel('Genes per cell', fontname='Times New Roman', fontsize=12)
    axes[5].set_ylabel('Mitochondrial %', fontname='Times New Roman', fontsize=12)
    axes[5].set_title('Genes vs MT%', fontname='Times New Roman', fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_figure_multiformat(fig, "QC_plots_detailed", out_dir)
    plt.close()


create_qc_plots(adata, out_dir)

# 过滤细胞和基因
print("过滤细胞...")
initial_cells = adata.n_obs
sc.pp.filter_cells(adata, min_genes=config.MIN_GENES)
adata = adata[adata.obs["pct_counts_mt"] < config.MAX_MT_PERCENT, :]

print("过滤基因...")
initial_genes = adata.n_vars
sc.pp.filter_genes(adata, min_cells=config.MIN_CELLS)

print(f"过滤结果: 细胞 {initial_cells} -> {adata.n_obs}, 基因 {initial_genes} -> {adata.n_vars}")

if adata.n_obs == 0:
    print("错误: 所有细胞都被过滤掉了!")
    exit()

# ================================
# Step 3: 归一化 & 高变基因选择
# ================================
print("\n>>> 归一化 & 特征选择...")

adata.raw = adata
sc.pp.normalize_total(adata, target_sum=config.TARGET_SUM)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata,
                            min_mean=config.HVG_MIN_MEAN,
                            max_mean=config.HVG_MAX_MEAN,
                            min_disp=config.HVG_MIN_DISP,
                            n_top_genes=config.N_TOP_GENES)
print(f"高变基因数量: {sum(adata.var.highly_variable)}")


def create_hvg_plot(adata, out_dir):
    """创建高变基因图 - 罗马字体无网格版本"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 应用干净样式
    for ax in axes:
        create_clean_axis(ax)

    var_df = adata.var

    # 均值vs离散度
    scatter1 = axes[0].scatter(var_df['means'], var_df['dispersions'],
                               c=['red' if x else 'blue' for x in var_df['highly_variable']],
                               s=1, alpha=0.6)
    axes[0].set_xlabel('Mean expression', fontname='Times New Roman', fontsize=12)
    axes[0].set_ylabel('Dispersion', fontname='Times New Roman', fontsize=12)
    axes[0].set_title('Mean vs Dispersion', fontname='Times New Roman', fontsize=16, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    # 均值vs标准化离散度
    scatter2 = axes[1].scatter(var_df['means'], var_df['dispersions_norm'],
                               c=['red' if x else 'blue' for x in var_df['highly_variable']],
                               s=1, alpha=0.6)
    axes[1].set_xlabel('Mean expression', fontname='Times New Roman', fontsize=12)
    axes[1].set_ylabel('Normalized dispersion', fontname='Times New Roman', fontsize=12)
    axes[1].set_title('Mean vs Normalized Dispersion', fontname='Times New Roman', fontsize=16, fontweight='bold')
    axes[1].set_xscale('log')

    # 高变基因分布
    hvg_counts = var_df['highly_variable'].value_counts()
    bars = axes[2].bar(['Non-HVG', 'HVG'], hvg_counts.values, color=['blue', 'red'], alpha=0.7)
    axes[2].set_ylabel('Number of genes', fontname='Times New Roman', fontsize=12)
    axes[2].set_title('Highly Variable Genes', fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 在柱状图上添加数值
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2., height + 5,
                     f'{int(height)}', ha='center', va='bottom', fontname='Times New Roman')

    plt.tight_layout()
    save_figure_multiformat(fig, "highly_variable_genes_custom", out_dir)
    plt.close()


create_hvg_plot(adata, out_dir)

adata = adata[:, adata.var.highly_variable]

# ================================
# Step 4: 降维分析
# ================================
print("\n>>> 降维分析...")

sc.pp.scale(adata, max_value=10, zero_center=False)
sc.tl.pca(adata, svd_solver="arpack")


def create_pca_variance_plot(adata, out_dir):
    """创建PCA方差解释图 - 罗马字体无网格版本"""
    variance_ratio = adata.uns['pca']['variance_ratio']
    n_pcs = len(variance_ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 应用干净样式
    create_clean_axis(ax1)
    create_clean_axis(ax2)

    # 方差解释比例
    bars = ax1.bar(range(1, n_pcs + 1), variance_ratio, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Principal Component', fontname='Times New Roman', fontsize=12)
    ax1.set_ylabel('Variance Ratio', fontname='Times New Roman', fontsize=12)
    ax1.set_title('Variance Ratio per PC', fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 累计方差解释
    cumulative_variance = np.cumsum(variance_ratio)
    ax2.plot(range(1, n_pcs + 1), cumulative_variance, 'b-', linewidth=2, alpha=0.7)
    ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% variance')
    ax2.set_xlabel('Number of PCs', fontname='Times New Roman', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Ratio', fontname='Times New Roman', fontsize=12)
    ax2.set_title('Cumulative Variance Explained', fontname='Times New Roman', fontsize=16, fontweight='bold')
    ax2.legend(prop={'family': 'Times New Roman'})

    plt.tight_layout()
    save_figure_multiformat(fig, "PCA_variance_custom", out_dir)
    plt.close()


create_pca_variance_plot(adata, out_dir)

n_pcs_actual = min(config.N_PCS, adata.n_vars - 1, adata.obsm['X_pca'].shape[1] - 1)
sc.pp.neighbors(adata, n_neighbors=config.N_NEIGHBORS, n_pcs=n_pcs_actual)

print("计算UMAP...")
sc.tl.umap(adata)

print("计算t-SNE...")
sc.tl.tsne(adata, use_rep='X_pca')


def create_embedding_plots(adata, out_dir):
    """创建降维可视化 - 罗马字体无网格版本"""
    # UMAP QC图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 应用干净样式
    for ax in axes:
        create_clean_axis(ax)

    # UMAP - 基因数
    scatter1 = axes[0].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                               c=adata.obs['n_genes_by_counts'], s=1, alpha=0.6, cmap='viridis')
    cbar1 = plt.colorbar(scatter1, ax=axes[0])
    cbar1.set_label('Genes per cell', fontname='Times New Roman', fontsize=11)
    axes[0].set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
    axes[0].set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
    axes[0].set_title('n_genes_by_counts', fontname='Times New Roman', fontsize=16, fontweight='bold')

    # UMAP - UMI总数
    scatter2 = axes[1].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                               c=adata.obs['total_counts'], s=1, alpha=0.6, cmap='viridis')
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Total counts', fontname='Times New Roman', fontsize=11)
    axes[1].set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
    axes[1].set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
    axes[1].set_title('total_counts', fontname='Times New Roman', fontsize=16, fontweight='bold')

    # UMAP - 线粒体百分比
    scatter3 = axes[2].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1],
                               c=adata.obs['pct_counts_mt'], s=1, alpha=0.6, cmap='viridis')
    cbar3 = plt.colorbar(scatter3, ax=axes[2])
    cbar3.set_label('MT%', fontname='Times New Roman', fontsize=11)
    axes[2].set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
    axes[2].set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
    axes[2].set_title('pct_counts_mt', fontname='Times New Roman', fontsize=16, fontweight='bold')

    axes[3].axis('off')

    plt.tight_layout()
    save_figure_multiformat(fig, "UMAP_QC_custom", out_dir)
    plt.close()


create_embedding_plots(adata, out_dir)

# ================================
# Step 5: 聚类
# ================================
print("\n>>> Leiden 聚类...")

for res in config.RESOLUTIONS:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}')

adata.obs['leiden'] = adata.obs['leiden_r0.5']
n_clusters = len(adata.obs['leiden'].cat.categories)
print(f"聚类数量 (resolution=0.5): {n_clusters}")

# ================================
# Step 6: 细胞类型注释
# ================================
print("\n>>> 细胞类型注释...")

# 乳腺癌标记基因字典
brca_markers = {
    'TLS': ['PLA2G2D', 'IGHA2', 'AC020656.1', 'LGALS2', 'CD6'],
    'B cells': ['CD19', 'MS4A1', 'CD79A', 'CD79B', 'IGHGP', 'IGHG1', 'IGHD', 'IGHM', 'HLA-DRA', 'TNFRSF17', 'CD38',
                'XBP1'],
    'T cells_CD4+': ['LGALS9', 'CD27', 'TNFRSF4', 'ICOS', 'TNFRSF9', 'PRF1', 'IFNG', 'GNLY', 'GZMA'],
    'T cells_CD8+': ['NKG7', 'CST7', 'GZMK', 'GZMB', 'CTLA4', 'HAVCR2', 'LAG3', 'PDCD1', 'TIGIT'],
    'NK cells': ['NKG7', 'GNLY', 'KLRD1', 'FCGR3A'],
    'Myeloid cells': ['CD14', 'FCGR3A', 'LYZ', 'MS4A7'],
    'Macrophages': ['CD68', 'CD163', 'MRC1', 'MSR1'],
    'Dendritic cells': ['FCER1A', 'CST3', 'CD1C', 'CLEC10A'],
    'Mast cells': ['CPA3', 'TPSAB1', 'TPSB2', 'MS4A2'],
    'Endothelial cells': ['PECAM1', 'VWF', 'CDH5', 'CLDN5'],
    'Fibroblasts': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM'],
    'Epithelial cells': ['EPCAM', 'KRT8', 'KRT18', 'KRT19'],
    'Cancer cells': ['EGFR', 'ERBB2', 'MUC1', 'KRT5', 'KRT17'],
    'Plasma cells': ['MZB1', 'JCHAIN', 'IGKC', 'IGLC2', 'SDC1'],
    'Proliferating cells': ['MKI67', 'TOP2A', 'PCNA', 'STMN1']
}

# 自动注释细胞类型
adata = annotate_cell_types(adata, brca_markers)

# ================================
# Step 6.1: 创建标记基因可视化
# ================================
print("\n>>> 创建标记基因可视化...")

# 创建标记基因表达图
create_marker_gene_plots(adata, brca_markers, out_dir, ncols=5)

# 创建细胞类型标记基因点图
create_celltype_marker_dotplot(adata, brca_markers, out_dir)

# 为每个细胞类型创建单独的图
create_celltype_specific_plots(adata, brca_markers, out_dir)

# ================================
# Step 6.2: 创建cluster与细胞类型对照表
# ================================
print("\n>>> 创建cluster与细胞类型对照表...")
create_cluster_celltype_table(adata, out_dir)


def create_clustering_plots(adata, out_dir):
    """创建聚类可视化 - 罗马字体无网格版本，包含细胞类型标签"""
    # 创建带图例的版本
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 应用干净样式
    create_clean_axis(axes[0])
    create_clean_axis(axes[1])

    categories = adata.obs['leiden'].cat.categories
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    # UMAP聚类
    for i, cluster in enumerate(categories):
        cluster_mask = adata.obs['leiden'] == cluster
        cell_type = adata.obs['cell_type'][cluster_mask].iloc[0]
        axes[0].scatter(adata.obsm['X_umap'][cluster_mask, 0],
                        adata.obsm['X_umap'][cluster_mask, 1],
                        c=[colors[i]], label=f'{cluster}: {cell_type}', s=8, alpha=0.7)

    axes[0].set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
    axes[0].set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
    axes[0].set_title('UMAP - Leiden Clustering with Cell Types',
                      fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 在UMAP图上添加细胞类型标签
    add_cluster_labels(axes[0], adata, embedding='X_umap', cluster_key='leiden',
                       cell_type_key='cell_type', fontsize=10, fontweight='bold', color='black')

    # t-SNE聚类
    for i, cluster in enumerate(categories):
        cluster_mask = adata.obs['leiden'] == cluster
        cell_type = adata.obs['cell_type'][cluster_mask].iloc[0]
        axes[1].scatter(adata.obsm['X_tsne'][cluster_mask, 0],
                        adata.obsm['X_tsne'][cluster_mask, 1],
                        c=[colors[i]], label=f'{cluster}: {cell_type}', s=8, alpha=0.7)

    axes[1].set_xlabel('t-SNE1', fontname='Times New Roman', fontsize=12)
    axes[1].set_ylabel('t-SNE2', fontname='Times New Roman', fontsize=12)
    axes[1].set_title('t-SNE - Leiden Clustering with Cell Types',
                      fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 在t-SNE图上添加细胞类型标签
    add_cluster_labels(axes[1], adata, embedding='X_tsne', cluster_key='leiden',
                       cell_type_key='cell_type', fontsize=10, fontweight='bold', color='black')

    # 添加图例
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   prop={'family': 'Times New Roman', 'size': 8},
                   title='Clusters & Cell Types', title_fontproperties={'family': 'Times New Roman', 'weight': 'bold'})
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   prop={'family': 'Times New Roman', 'size': 8},
                   title='Clusters & Cell Types', title_fontproperties={'family': 'Times New Roman', 'weight': 'bold'})

    plt.tight_layout()
    save_figure_multiformat(fig, "clustering_results_with_cell_types", out_dir)
    plt.close()

    # 创建无图例但带大标签的版本
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    create_clean_axis(axes[0])
    create_clean_axis(axes[1])

    # UMAP聚类（无图例）
    for i, cluster in enumerate(categories):
        cluster_mask = adata.obs['leiden'] == cluster
        axes[0].scatter(adata.obsm['X_umap'][cluster_mask, 0],
                        adata.obsm['X_umap'][cluster_mask, 1],
                        c=[colors[i]], s=5, alpha=0.7)

    axes[0].set_xlabel('UMAP1', fontname='Times New Roman', fontsize=12)
    axes[0].set_ylabel('UMAP2', fontname='Times New Roman', fontsize=12)
    axes[0].set_title('UMAP - Cell Type Annotation',
                      fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 在UMAP图上添加更大的细胞类型标签
    add_cluster_labels(axes[0], adata, embedding='X_umap', cluster_key='leiden',
                       cell_type_key='cell_type', fontsize=12, fontweight='bold', color='black')

    # t-SNE聚类（无图例）
    for i, cluster in enumerate(categories):
        cluster_mask = adata.obs['leiden'] == cluster
        axes[1].scatter(adata.obsm['X_tsne'][cluster_mask, 0],
                        adata.obsm['X_tsne'][cluster_mask, 1],
                        c=[colors[i]], s=5, alpha=0.7)

    axes[1].set_xlabel('t-SNE1', fontname='Times New Roman', fontsize=12)
    axes[1].set_ylabel('t-SNE2', fontname='Times New Roman', fontsize=12)
    axes[1].set_title('t-SNE - Cell Type Annotation',
                      fontname='Times New Roman', fontsize=16, fontweight='bold')

    # 在t-SNE图上添加更大的细胞类型标签
    add_cluster_labels(axes[1], adata, embedding='X_tsne', cluster_key='leiden',
                       cell_type_key='cell_type', fontsize=12, fontweight='bold', color='black')

    plt.tight_layout()
    save_figure_multiformat(fig, "cell_type_annotation_only", out_dir)
    plt.close()


try:
    create_clustering_plots(adata, out_dir)
    print("聚类可视化生成成功")
except Exception as e:
    print(f"生成聚类可视化时出错: {e}")
    import traceback

    traceback.print_exc()

# ================================
# Step 7: Marker基因识别
# ================================
print("\n>>> 计算差异基因...")

sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", use_raw=False)


def create_marker_plots(adata, out_dir):
    """创建marker基因可视化 - 罗马字体无网格版本，使用细胞类型名称"""
    try:
        marker_df = sc.get.rank_genes_groups_df(adata, group=None)

        # 获取每个cluster的主要细胞类型
        cluster_to_celltype = {}
        for cluster in adata.obs['leiden'].cat.categories:
            cluster_mask = adata.obs['leiden'] == cluster
            if 'cell_type' in adata.obs.columns:
                # 获取该cluster中最常见的细胞类型
                cell_type_counts = adata.obs['cell_type'][cluster_mask].value_counts()
                if not cell_type_counts.empty:
                    main_cell_type = cell_type_counts.index[0]
                    # 简化过长的细胞类型名称
                    if len(main_cell_type) > 20:
                        main_cell_type = main_cell_type[:20] + "..."
                    cluster_to_celltype[cluster] = f"{cluster}: {main_cell_type}"
                else:
                    cluster_to_celltype[cluster] = str(cluster)
            else:
                cluster_to_celltype[cluster] = str(cluster)

        # 收集每个cluster的top marker基因
        top_markers = {}
        for cluster in adata.obs['leiden'].cat.categories:
            cluster_markers = marker_df[marker_df['group'] == cluster].head(5)
            top_markers[cluster] = cluster_markers['names'].tolist()

        # 获取所有top marker基因
        all_top_markers = list(set([gene for markers in top_markers.values() for gene in markers]))

        if len(all_top_markers) > 0:
            # 创建表达矩阵
            expr_matrix = pd.DataFrame(index=adata.obs['leiden'].cat.categories,
                                       columns=all_top_markers)
            pct_matrix = pd.DataFrame(index=adata.obs['leiden'].cat.categories,
                                      columns=all_top_markers)

            # 计算每个cluster中每个基因的平均表达量和表达细胞比例
            for cluster in expr_matrix.index:
                cluster_mask = adata.obs['leiden'] == cluster
                cluster_data = adata[cluster_mask, :]

                for gene in all_top_markers:
                    if gene in adata.var_names:
                        if scipy.sparse.issparse(cluster_data[:, gene].X):
                            expr_values = cluster_data[:, gene].X.toarray().flatten()
                        else:
                            expr_values = cluster_data[:, gene].X.flatten()

                        expr_matrix.loc[cluster, gene] = np.mean(expr_values)
                        pct_matrix.loc[cluster, gene] = np.mean(expr_values > 0)
                    else:
                        expr_matrix.loc[cluster, gene] = 0
                        pct_matrix.loc[cluster, gene] = 0

            # 创建图形
            fig, ax = plt.subplots(figsize=(max(10, len(all_top_markers)),
                                            max(8, len(expr_matrix.index))))
            create_clean_axis(ax)

            # 创建点图
            for i, cluster in enumerate(expr_matrix.index):
                for j, gene in enumerate(expr_matrix.columns):
                    # 点的大小表示表达该基因的细胞比例
                    size = pct_matrix.loc[cluster, gene] * 500
                    # 颜色表示平均表达量
                    color_val = expr_matrix.loc[cluster, gene]

                    ax.scatter(j, i, s=size, c=[color_val], cmap='viridis',
                               vmin=0, vmax=expr_matrix.values.max(),
                               alpha=0.7, edgecolors='black', linewidth=0.5)

            # 设置x轴标签（基因名称）
            ax.set_xticks(range(len(expr_matrix.columns)))
            ax.set_xticklabels(expr_matrix.columns, rotation=45, ha='right',
                               fontname='Times New Roman', fontsize=10)

            # 设置y轴标签（使用细胞类型名称）
            y_labels = [cluster_to_celltype.get(cluster, str(cluster))
                        for cluster in expr_matrix.index]
            ax.set_yticks(range(len(expr_matrix.index)))
            ax.set_yticklabels(y_labels, fontname='Times New Roman', fontsize=10)

            ax.set_xlabel('Genes', fontname='Times New Roman', fontsize=12)
            ax.set_ylabel('Clusters with Cell Types', fontname='Times New Roman', fontsize=12)
            ax.set_title('Top Marker Genes Expression by Cell Type',
                         fontname='Times New Roman', fontsize=16, fontweight='bold')

            # 添加颜色条
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                       norm=plt.Normalize(0, expr_matrix.values.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Mean Expression', fontname='Times New Roman', fontsize=11)
            for label in cbar.ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(9)

            plt.tight_layout()
            save_figure_multiformat(fig, "marker_genes_dotplot_custom", out_dir)
            plt.close()

            # 另外创建一个更简洁的版本，只显示细胞类型名称
            fig2, ax2 = plt.subplots(figsize=(max(10, len(all_top_markers)),
                                              max(8, len(expr_matrix.index))))
            create_clean_axis(ax2)

            # 创建点图
            for i, cluster in enumerate(expr_matrix.index):
                for j, gene in enumerate(expr_matrix.columns):
                    size = pct_matrix.loc[cluster, gene] * 500
                    color_val = expr_matrix.loc[cluster, gene]

                    ax2.scatter(j, i, s=size, c=[color_val], cmap='viridis',
                                vmin=0, vmax=expr_matrix.values.max(),
                                alpha=0.7, edgecolors='black', linewidth=0.5)

            # 设置x轴标签
            ax2.set_xticks(range(len(expr_matrix.columns)))
            ax2.set_xticklabels(expr_matrix.columns, rotation=45, ha='right',
                                fontname='Times New Roman', fontsize=10)

            # 设置y轴标签 - 只显示细胞类型名称（不带cluster编号）
            celltype_labels = []
            for cluster in expr_matrix.index:
                if 'cell_type' in adata.obs.columns:
                    cluster_mask = adata.obs['leiden'] == cluster
                    cell_type_counts = adata.obs['cell_type'][cluster_mask].value_counts()
                    if not cell_type_counts.empty:
                        main_cell_type = cell_type_counts.index[0]
                        # 简化过长的细胞类型名称
                        if len(main_cell_type) > 15:
                            main_cell_type = main_cell_type[:15] + "..."
                        celltype_labels.append(main_cell_type)
                    else:
                        celltype_labels.append(str(cluster))
                else:
                    celltype_labels.append(str(cluster))

            ax2.set_yticks(range(len(expr_matrix.index)))
            ax2.set_yticklabels(celltype_labels, fontname='Times New Roman', fontsize=10)

            ax2.set_xlabel('Genes', fontname='Times New Roman', fontsize=12)
            ax2.set_ylabel('Cell Types', fontname='Times New Roman', fontsize=12)
            ax2.set_title('Top Marker Genes Expression by Cell Type',
                          fontname='Times New Roman', fontsize=16, fontweight='bold')

            # 添加颜色条
            sm2 = plt.cm.ScalarMappable(cmap='viridis',
                                        norm=plt.Normalize(0, expr_matrix.values.max()))
            sm2.set_array([])
            cbar2 = plt.colorbar(sm2, ax=ax2)
            cbar2.set_label('Mean Expression', fontname='Times New Roman', fontsize=11)
            for label in cbar2.ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(9)

            plt.tight_layout()
            save_figure_multiformat(fig2, "marker_genes_dotplot_celltypes_only", out_dir)
            plt.close()

            print(f"成功创建marker基因点图，包含 {len(all_top_markers)} 个基因和 {len(expr_matrix.index)} 个细胞类型")

    except Exception as e:
        print(f"创建marker基因图时出错: {e}")
        import traceback
        traceback.print_exc()


create_marker_plots(adata, out_dir)

# 保存差异基因结果
try:
    marker_df = sc.get.rank_genes_groups_df(adata, group=None)
    marker_df.to_csv(os.path.join(out_dir, "marker_genes_detailed.csv"), index=False)
    print(f"成功保存差异基因结果，共 {len(marker_df)} 行")
except Exception as e:
    print(f"保存差异基因结果时出错: {e}")

# ================================
# Step 8: Python细胞通讯分析 - 使用细胞类型名称
# ================================
print("\n>>> 开始细胞通讯分析（使用细胞类型名称）...")

# 运行优化的细胞通讯分析（使用细胞类型名称）
comm_success = run_optimized_communication_analysis_with_names(adata, out_dir, cluster_key='leiden')

# ================================
# Step 9: 保存所有结果
# ================================
print("\n>>> 保存所有分析结果...")

# 保存AnnData对象
try:
    adata.write(os.path.join(out_dir, "scRNA_analysis_roman_font.h5ad"))
    print("成功保存AnnData对象")
except Exception as e:
    print(f"保存AnnData对象时出错: {e}")

# 保存聚类结果
try:
    clustering_cols = [f'leiden_r{res}' for res in config.RESOLUTIONS] + ['leiden', 'cell_type']
    clustering_results = adata.obs[clustering_cols].copy()
    clustering_results.to_csv(os.path.join(out_dir, "clustering_results.csv"))
    print("成功保存聚类结果")
except Exception as e:
    print(f"保存聚类结果时出错: {e}")

# 保存QC指标
try:
    qc_metrics = adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']].copy()
    qc_metrics.to_csv(os.path.join(out_dir, "qc_metrics.csv"))
    print("成功保存QC指标")
except Exception as e:
    print(f"保存QC指标时出错: {e}")

# 保存高变基因信息
try:
    hvg_info = adata.var[['highly_variable', 'means', 'dispersions', 'dispersions_norm']].copy()
    hvg_info.to_csv(os.path.join(out_dir, "highly_variable_genes_info.csv"))
    print("成功保存高变基因信息")
except Exception as e:
    print(f"保存高变基因信息时出错: {e}")

# 保存PCA坐标
try:
    pca_coords = pd.DataFrame(adata.obsm['X_pca'][:, :10],
                              index=adata.obs_names,
                              columns=[f'PC{i + 1}' for i in range(10)])
    pca_coords.to_csv(os.path.join(out_dir, "pca_coordinates.csv"))
    print("成功保存PCA坐标")
except Exception as e:
    print(f"保存PCA坐标时出错: {e}")

# 保存UMAP和t-SNE坐标
try:
    embedding_coords = pd.DataFrame({
        'UMAP1': adata.obsm['X_umap'][:, 0],
        'UMAP2': adata.obsm['X_umap'][:, 1],
        'tSNE1': adata.obsm['X_tsne'][:, 0],
        'tSNE2': adata.obsm['X_tsne'][:, 1]
    }, index=adata.obs_names)
    embedding_coords.to_csv(os.path.join(out_dir, "embedding_coordinates.csv"))
    print("成功保存降维坐标")
except Exception as e:
    print(f"保存降维坐标时出错: {e}")

# 生成详细分析报告
report = f"""
单细胞RNA测序分析报告 (罗马字体优化版)
==================================

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据路径: {base_dir}

数据概览:
---------
原始数据: {qc_before['cells']} 细胞, {qc_before['genes']} 基因
过滤后: {adata.n_obs} 细胞, {adata.n_vars} 基因
过滤比例: {((initial_cells - adata.n_obs) / initial_cells * 100):.1f}% 细胞被过滤

质量控制:
---------
过滤前中位数基因数: {qc_before['median_genes']:.0f}
过滤前中位数UMI: {qc_before['median_umi']:.0f}
过滤前中位数线粒体%: {qc_before['median_mt']:.2f}%

聚类分析:
---------
聚类数 (resolution=0.5): {n_clusters}
高变基因数: {sum(adata.var.highly_variable)}

细胞类型注释:
-------------
{chr(10).join([f"Cluster {cluster}: {adata.obs[adata.obs['leiden'] == cluster]['cell_type'].iloc[0]}" for cluster in sorted(adata.obs['leiden'].cat.categories)])}

标记基因可视化:
--------------
1. 按细胞类型分组的标记基因表达图
2. 细胞类型标记基因点图 (显示cluster编号和细胞类型)
3. 每个细胞类型的详细分析图

细胞类型对照表:
---------------
已生成cluster_celltype_mapping.csv和cluster_celltype_table可视化表格

细胞通讯分析 (使用细胞类型名称):
--------------------------------
Python通讯分析: {'已完成' if comm_success else '未完成/出错'}
分析方法: 基于配体-受体表达的通讯评分
输出格式: 所有热图、气泡图、柱状图均使用细胞类型名称而非数字编号

主要通讯可视化:
---------------
1. 基于细胞类型的通讯网络热图 (communication_network_by_celltype)
2. 基于细胞类型的配体-受体气泡图 (ligand_receptor_bubble_with_names)
3. 基于细胞类型的信号通路分析 (pathway_activity_by_name)
4. 各重要通路的细胞类型热图 (pathway_*_by_celltype)
5. 基于细胞类型的细胞角色分析 (cell_communication_roles_by_name)
6. 基于细胞类型的相互作用类型分析 (interaction_type_analysis_by_name)

图像优化:
---------
- 所有通讯分析图中使用细胞类型名称而非数字编号
- 细胞类型标签格式: "cluster编号\\n细胞类型名称"
- 长细胞类型名称自动截断显示
- 所有图像使用罗马字体(Times New Roman)
- 无网格线设计
- 高清分辨率输出 (600 DPI)

输出文件:
--------
- 基于细胞类型的通讯分析目录: communication_analysis_with_names/
- 包含细胞类型名称的结果文件: *_with_celltypes.csv
- 细胞类型通讯热图: communication_network_by_celltype.jpg/pdf
- 细胞类型配体-受体气泡图: ligand_receptor_bubble_with_names.jpg/pdf
- 细胞类型信号通路图: pathway_activity_by_name.jpg/pdf
- 各通路细胞类型热图: pathway_*_by_celltype.jpg/pdf
- 细胞类型角色分析: cell_communication_roles_by_name.jpg/pdf
- 相互作用类型分析: interaction_type_analysis_by_name.jpg/pdf
- 所有基础分析结果文件
- AnnData对象: scRNA_analysis_roman_font.h5ad

特别说明:
---------
所有通讯分析可视化已优化，使用细胞类型名称替代数字编号，使结果更直观易懂。
对于Angiogenesis、Chemokine、ECM、Notch等重要通路，均已生成基于细胞类型的热图。

结果保存位置: {out_dir}
"""

print(report)
with open(os.path.join(out_dir, "analysis_report_roman_font_with_celltypes.txt"), "w", encoding='utf-8') as f:
    f.write(report)

# 保存配置信息
config_info = pd.DataFrame([{k: v for k, v in vars(config).items() if not k.startswith('_')}])
config_info.to_csv(os.path.join(out_dir, "analysis_configuration.csv"), index=False)

print(f">>> 分析完成! 所有结果保存在: {out_dir}")
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"生成的图像格式: JPG (用于查看) + PDF (用于出版)")
print(f"图像风格: 罗马字体 + 无网格线 + 高清分辨率 + 细胞类型名称")
print(f"通讯分析优化: 所有图像使用细胞类型名称而非数字编号")
print(f"重要通路热图: Angiogenesis、Chemokine、ECM、Notch等通路均已生成基于细胞类型的可视化")
print(f"输出目录: communication_analysis_with_names/ (包含所有基于细胞类型的分析结果)")
