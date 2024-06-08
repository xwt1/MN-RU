import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(root_path):
    # CSV文件路径和保存图片的路径
    csv_file = root_path + '/output/figure_1/compare_queryTime_and_deleteUpdateTime'
    output_path = root_path + '/output/figure_1/figure.png'

    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 将recall转换为百分比
    df['recall'] = df['recall'] * 100

    # 设置图表尺寸
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 设置条形图宽度
    bar_width = 0.2

    # 设置每个条形图的位置
    index = np.arange(len(df['dataset_name']))

    # 绘制query_time和delete_update_time条形图在左轴
    ax1.bar(index, df['query_time'], bar_width, label='Query Time (s)', color='b', align='center')
    ax1.bar(index + bar_width, df['delete_update_time'], bar_width, label='Delete/Update Time (s)', color='g', align='center')

    # 设置左轴标签和标题
    ax1.set_xlabel('Dataset Name')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Query and Delete/Update Time with Recall for Different Datasets')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(df['dataset_name'])

    # 创建一个新的坐标轴，右侧显示recall
    ax2 = ax1.twinx()

    # 绘制recall条形图在右轴
    ax2.bar(index + 2 * bar_width, df['recall'], bar_width, label='Recall (%)', color='r', align='center')

    # 设置右轴标签
    ax2.set_ylabel('Recall (%)')

    # 合并图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    # 保存图表到指定路径
    plt.savefig(output_path)

    # 显示图表
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate bar chart from CSV data.')
    parser.add_argument('root_path', type=str, help='Root path for input CSV and output image')
    args = parser.parse_args()
    main(args.root_path)