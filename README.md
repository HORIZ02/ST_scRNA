# ST_scRNA
analysis_results2_HER2_cell-cell_name_TLS10x/
├── communication_analysis_with_names/     # 细胞通讯分析结果
│   ├── communication_network_by_celltype.jpg/pdf
│   ├── ligand_receptor_bubble_with_names.jpg/pdf
│   ├── pathway_activity_by_name.jpg/pdf
│   ├── cell_communication_roles_by_name.jpg/pdf
│   └── *.csv (各种通讯分析数据表)
├── QC_plots_detailed.jpg/pdf              # 质量控制图
├── highly_variable_genes_custom.jpg/pdf   # 高变基因图
├── clustering_results_with_cell_types.jpg/pdf # 聚类结果
├── marker_genes_summary_umap.jpg/pdf      # 标记基因表达图
├── celltype_marker_dotplot.jpg/pdf        # 细胞类型标记基因点图
├── cluster_celltype_mapping.csv           # 聚类-细胞类型映射表
├── scRNA_analysis_roman_font.h5ad         # 完整的AnnData对象
└── analysis_report_roman_font_with_celltypes.txt # 分析报告
conda create -n scrna_analysis python=3.9
conda activate scrna_analysis
pip install scanpy anndata pandas numpy scipy matplotlib seaborn
