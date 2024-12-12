import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# 添加背景填充函数
def add_background(ax, text, orientation, color, fontsize=13):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=color, alpha=0.33))
    rotation = 90 if orientation == 'vertical' else 0
    ax.text(0.5, 0.5, text, fontsize=fontsize, fontweight='bold', ha='center', va='center',
            rotation=rotation, transform=ax.transAxes)
    ax.axis('off')

# 配置绘图轴
def configure_axes(ax):
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(3)
    ax.tick_params(axis='both', which='both', width=3, labelsize=14)

if __name__ == "__main__":
    # 设置全局样式
    plt.rcParams.update({
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18
    })

    # 定义数据集和工作负载
    workloads = ["Full Dataset", "Random"]
    datasets = ["Gist", "ImageNet"]

    # 定义 CSV 路径字典（请根据实际情况修改）
    csv_paths = {
        "Full Dataset": {
            "Gist": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/edge_connected_replaced_update9_end_recall_gist_1M.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/replaced_update_end_recall_gist_1M.csv",
            },
            "ImageNet": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/edge_connected_replaced_update9_end_recall_imageNet_2M.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/replaced_update_end_recall_imageNet_2M.csv",
            }
        },
        "Random": {
            "Gist": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/edge_connected_9_end_recall_gist_1M.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/replaced_update_end_recall_gist_1M.csv",
            },
            "ImageNet": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/edge_connected_9_end_recall_imageNet_2M.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/replaced_update_end_recall_imageNet_2M.csv",
            }
        }
    }

    # 创建图形和网格
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 收集全局图例句柄和标签
    line_handles = []
    line_labels = []

    # 绘制 Full Dataset 和 Random 的 Gist 和 ImageNet
    for i, workload in enumerate(workloads):
        for j, dataset in enumerate(datasets):
            ax = fig.add_subplot(gs[i, j])
            configure_axes(ax)
            ax.set_title(f"{workload} - {dataset}", fontsize=16, fontweight='bold')

            if dataset in csv_paths[workload]:
                # 绘制该数据集的三条曲线
                for label, path in csv_paths[workload][dataset].items():
                    try:
                        data_csv = pd.read_csv(path)
                    except FileNotFoundError:
                        ax.text(0.5, 0.5, "File Not Found", ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                        continue

                    x_data = data_csv["query_time"] * 1000
                    y_data = data_csv["recall"]
                    line, = ax.plot(x_data, y_data, label=label, linewidth=2.5)
                    if label not in line_labels:
                        line_handles.append(line)
                        line_labels.append(label)
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

    # 调整全局标签位置
    fig.text(0.5, 0.06, 'Query Time (ms)', ha='center', fontsize=18, fontweight='bold')
    fig.text(0.09, 0.5, 'Recall (%)', va='center', rotation='vertical', fontsize=18, fontweight='bold')

    # 全局图例
    fig.legend(line_handles, line_labels, loc='lower center', bbox_to_anchor=(0.5, 0.015), ncol=3, fontsize=12)

    # 保存为 PDF
    output_path = "output_plot_recall_subset.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"图像已保存为 PDF 文件: {output_path}")
    plt.show()
