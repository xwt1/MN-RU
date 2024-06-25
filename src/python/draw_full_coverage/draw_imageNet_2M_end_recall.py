import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_recall_vs_query_time(csv_files, colors, markers, labels, output_path):
    # 设置图表尺寸和DPI
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    for csv_file, color, marker, label in zip(csv_files, colors, markers, labels):
        # 检查CSV文件是否存在
        if not os.path.exists(csv_file):
            print(f"CSV file '{csv_file}' not found.")
            continue

        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 绘制折线图，每隔5个点绘制一个点，点的大小为5
        ax.plot(df['query_time'], df['recall'], linestyle='-', color=color, marker=marker, label=label, markevery=5, markersize=5)

    # 设置图表标签和标题
    ax.set_xlabel('Query Time')
    ax.set_ylabel('Recall')
    ax.set_title('Recall vs Query Time')
    ax.legend(prop={'size': 10, 'weight': 'bold'})
    ax.grid(True, linestyle='--', color='grey', linewidth=0.5)

    # 保存图表
    fig.savefig(output_path, bbox_inches='tight')

    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate line chart from CSV data for recall vs query time.')
    parser.add_argument('root_path', type=str, help='Root path for input CSV files and output images')
    args = parser.parse_args()

    # 定义CSV文件的根目录和图像的输出路径
    csv_dir = os.path.join(args.root_path, 'output', 'full_coverage', 'imageNet')
    output_path = os.path.join(args.root_path, 'output', 'full_coverage', 'imageNet', 'full_coverage_imageNet_2M_end_recall.png')

    # 手动编码的CSV文件名列表
    csv_files = [
        'edge_connected_replaced_update9_end_recall_imageNet_2M.csv',
        'edge_connected_replaced_update10_end_recall_imageNet_2M.csv',
        'replaced_update_end_recall_imageNet_2M.csv'
    ]
    # 手动编码的颜色和点的形状
    colors = ['b', 'm', 'c']
    markers = ['D', '^', '*']
    # 手动编码的图例标签
    labels = [
        'MN-RU γ',
        'MN-THN-RU',
        'HNSW-RU'
    ]

    csv_file_paths = [os.path.join(csv_dir, csv_file) for csv_file in csv_files]

    plot_recall_vs_query_time(csv_file_paths, colors, markers, labels, output_path)
