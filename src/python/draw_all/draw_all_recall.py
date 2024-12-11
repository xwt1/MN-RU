# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Rectangle
# from matplotlib.gridspec import GridSpec
#
# # 添加背景填充函数
# def add_background(ax, text, orientation, color, fontsize=13):
#     ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=color, alpha=0.33))
#     rotation = 90 if orientation == 'vertical' else 0
#     ax.text(0.5, 0.5, text, fontsize=fontsize, fontweight='bold', ha='center', va='center',
#             rotation=rotation, transform=ax.transAxes)
#     ax.axis('off')
#
# # 配置绘图轴
# def configure_axes(ax):
#     ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
#     for spine in ['top', 'right']:
#         ax.spines[spine].set_visible(False)
#     for spine in ['left', 'bottom']:
#         ax.spines[spine].set_linewidth(3)
#     ax.tick_params(axis='both', which='both', width=3, labelsize=14)
#
# if __name__ == "__main__":
#     # 设置全局样式
#     plt.rcParams.update({
#         'axes.labelsize': 20,
#         'xtick.labelsize': 18,
#         'ytick.labelsize': 18
#     })
#
#     # 定义数据集和工作负载
#     workloads = ["Full Dataset", "Random", "Incremental"]
#     datasets = ["Gist", "ImageNet", "Sift", "Sift100M"]
#     incremental_dataset = ["Sift2M"]
#
#     # 定义 CSV 路径字典
#     csv_paths = {
#         "Full Dataset": {
#             "Gist": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/faiss_end_recall_gist_1M.csv"
#             },
#             "ImageNet": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/faiss_end_recall_imageNet_2M.csv"
#             },
#             "Sift": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/sift/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/sift/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/sift/faiss_end_recall_sift_1M.csv"
#             },
#             "Sift100M": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/full_coverage/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/full_coverage/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/full_coverage/faiss_end_recall_sift_100M.csv"
#             }
#         },
#         "Random": {
#             "Gist": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/faiss_end_recall_gist_1M.csv"
#             },
#             "ImageNet": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/faiss_end_recall_imageNet_2M.csv"
#             },
#             "Sift": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/sift/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/sift/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/sift/faiss_end_recall_sift_1M.csv"
#             },
#             "Sift100M": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/random/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/random/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/random/faiss_end_recall_sift_100M.csv"
#             }
#         },
#         "Incremental": {
#             "Sift2M": {
#                 "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/new_insert/sift_2M/edge_connected_replaced_update9_end_recall.csv",
#                 "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/new_insert/sift_2M/replaced_update_end_recall.csv",
#                 "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/new_insert/sift_2M/faiss_end_recall_sift_2M.csv"
#             }
#         }
#     }
#
#     # 创建图形和网格
#     fig = plt.figure(figsize=(18, 12))
#     gs = GridSpec(5, 5, figure=fig, width_ratios=[0.15, 1, 1, 1, 1], height_ratios=[0.2, 1, 1, 0.2, 1],
#                   hspace=0.3, wspace=0.3)
#
#     # 第一行: 数据集的背景填充
#     for j, dataset in enumerate(datasets):
#         ax_bg = fig.add_subplot(gs[0, j + 1])
#         add_background(ax_bg, dataset, orientation='horizontal', color='lightblue')
#
#     # 添加 workloads 背景填充
#     for i, workload in enumerate(workloads):
#         # 恢复为原程序的对齐方式
#         if i == 2:
#             ax_bg = fig.add_subplot(gs[i + 2, 0])  # Incremental放在gs[4,0]
#         else:
#             ax_bg = fig.add_subplot(gs[i + 1, 0])
#         add_background(ax_bg, workload, orientation='vertical', color='lightgreen')
#
#     # 第三行背景填充（Sift 2M）
#     ax_bg = fig.add_subplot(gs[3, 1:])
#     add_background(ax_bg, text="Sift 2M", orientation='horizontal', color='lightblue')
#
#     # 绘制 Full Dataset 和 Random
#     for i, workload in enumerate(workloads):
#         if workload in ["Full Dataset", "Random"]:
#             # Full Dataset行: i=0 -> gs[1, :]
#             # Random行: i=1 -> gs[2, :]
#             row = i + 1
#             for j, dataset in enumerate(datasets):
#                 ax = fig.add_subplot(gs[row, j + 1])
#                 configure_axes(ax)
#                 if dataset in csv_paths[workload]:
#                     # 绘制该数据集的三条曲线
#                     for label, path in csv_paths[workload][dataset].items():
#                         data_csv = pd.read_csv(path)
#                         x_data = data_csv["query_time"] * 1000
#                         y_data = data_csv["recall"]  # 转换为ms
#                         ax.plot(x_data, y_data, label=label, linewidth=2.5)
#                     ax.legend(fontsize=12)
#                 else:
#                     # 如果没有数据就空一块或使用占位数据
#                     ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
#                     ax.axis('off')
#
#         elif workload == "Incremental":
#             # Incremental只有Sift2M数据集，需要放在最后一行 (第5行, gs[4, 1:])
#             ax_last = fig.add_subplot(gs[4, 1:])
#             configure_axes(ax_last)
#             dataset = "Sift2M"
#             if dataset in csv_paths[workload]:
#                 for label, path in csv_paths[workload][dataset].items():
#                     data_csv = pd.read_csv(path)
#                     x_data = data_csv["query_time"] * 1000
#                     y_data = data_csv["recall"]  # 转换为ms
#                     ax_last.plot(x_data, y_data, label=label, linewidth=2.5)
#                 ax_last.legend(fontsize=12)
#             else:
#                 ax_last.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax_last.transAxes)
#                 ax_last.axis('off')
#
#     # 添加整体 X 和 Y 坐标轴标签
#     fig.text(0.5, 0.06, 'Query Time (ms)', ha='center', fontsize=18, fontweight='bold')
#     fig.text(0.09, 0.5, 'Recall (%)', va='center', rotation='vertical', fontsize=18, fontweight='bold')
#
#     # 保存为 PDF
#     output_path = "output_plot_recall.pdf"
#     plt.savefig(output_path, format="pdf", bbox_inches="tight")
#     print(f"图像已保存为 PDF 文件: {output_path}")
#     plt.show()

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
    workloads = ["Full Dataset", "Random", "Incremental"]
    datasets = ["Gist", "ImageNet", "Sift", "Sift100M"]
    incremental_dataset = ["Sift2M"]

    # 定义 CSV 路径字典（请根据实际情况修改）
    csv_paths = {
        "Full Dataset": {
            "Gist": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/gist/faiss_end_recall_gist_1M.csv"
            },
            "ImageNet": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/imageNet/faiss_end_recall_imageNet_2M.csv"
            },
            "Sift": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/sift/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/sift/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/full_coverage/sift/faiss_end_recall_sift_1M.csv"
            },
            "Sift100M": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/full_coverage/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/full_coverage/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/full_coverage/faiss_end_recall_sift_100M.csv"
            }
        },
        "Random": {
            "Gist": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/gist/faiss_end_recall_gist_1M.csv"
            },
            "ImageNet": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/imageNet/faiss_end_recall_imageNet_2M.csv"
            },
            "Sift": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/sift/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/sift/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/random/sift/faiss_end_recall_sift_1M.csv"
            },
            "Sift100M": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/random/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/random/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/sift_200M/random/faiss_end_recall_sift_100M.csv"
            }
        },
        "Incremental": {
            "Sift2M": {
                "Mint": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/new_insert/sift_2M/edge_connected_replaced_update9_end_recall.csv",
                "HNSW-RU": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/new_insert/sift_2M/replaced_update_end_recall.csv",
                "IVF-FLAT": "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement/output/new_insert/sift_2M/faiss_end_recall_sift_2M.csv"
            }
        }
    }

    # 创建图形和网格
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(5, 5, figure=fig, width_ratios=[0.15, 1, 1, 1, 1], height_ratios=[0.2, 1, 1, 0.2, 1],
                  hspace=0.3, wspace=0.3)

    # 第一行: 数据集的背景填充
    for j, dataset in enumerate(datasets):
        ax_bg = fig.add_subplot(gs[0, j + 1])
        add_background(ax_bg, dataset, orientation='horizontal', color='lightblue')

    # 添加 workloads 背景填充
    for i, workload in enumerate(workloads):
        if i == 2:
            ax_bg = fig.add_subplot(gs[i + 2, 0])  # Incremental在第5行的左边列
        else:
            ax_bg = fig.add_subplot(gs[i + 1, 0])
        add_background(ax_bg, workload, orientation='vertical', color='lightgreen')

    # 第三行背景填充（Sift 2M）
    ax_bg = fig.add_subplot(gs[3, 1:])
    add_background(ax_bg, text="Sift 2M", orientation='horizontal', color='lightblue')

    # 收集全局图例句柄和标签
    line_handles = []
    line_labels = []

    # 绘制 Full Dataset 和 Random，以及 Incremental
    for i, workload in enumerate(workloads):
        if workload in ["Full Dataset", "Random"]:
            # Full Dataset行: i=0 -> gs[1, :]
            # Random行: i=1 -> gs[2, :]
            row = i + 1
            for j, dataset in enumerate(datasets):
                ax = fig.add_subplot(gs[row, j + 1])
                configure_axes(ax)
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

        elif workload == "Incremental":
            # Incremental只有Sift2M数据集需要放在最后一行 (gs[4,1:])
            ax_last = fig.add_subplot(gs[4, 1:])
            configure_axes(ax_last)
            dataset = "Sift2M"
            if dataset in csv_paths[workload]:
                for label, path in csv_paths[workload][dataset].items():
                    try:
                        data_csv = pd.read_csv(path)
                    except FileNotFoundError:
                        ax_last.text(0.5, 0.5, "File Not Found", ha='center', va='center', transform=ax_last.transAxes)
                        ax_last.axis('off')
                        continue

                    x_data = data_csv["query_time"] * 1000
                    y_data = data_csv["recall"]
                    line, = ax_last.plot(x_data, y_data, label=label, linewidth=2.5)
                    if label not in line_labels:
                        line_handles.append(line)
                        line_labels.append(label)
            else:
                ax_last.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax_last.transAxes)
                ax_last.axis('off')

    # 调整全局标签位置，让 X 轴标签稍微靠上一点
    fig.text(0.5, 0.07, 'Query Time (ms)', ha='center', fontsize=18, fontweight='bold')
    fig.text(0.09, 0.5, 'Recall (%)', va='center', rotation='vertical', fontsize=18, fontweight='bold')

    # 全局图例放在下方
    # 可以根据需要微调 bbox_to_anchor 的纵坐标（如 -0.05 改为 -0.03）来控制与主图的距离
    fig.legend(line_handles, line_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)

    # 减少底部空间，避免 X 轴标签离图太远
    fig.subplots_adjust(bottom=0.12)

    # 保存为 PDF
    output_path = "output_plot_recall.pdf"
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    print(f"图像已保存为 PDF 文件: {output_path}")
    plt.show()
