# import hnswlib
# import numpy as np
# import os
# import time
# import matplotlib.pyplot as plt


# def load_fvecs(filename):
#     """Read .fvecs file format"""
#     with open(filename, 'rb') as f:
#         d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # Dimensionality
#         n = int(len(f.read()) / (d + 1) / 4)
#         f.seek(0)
#         data = np.fromfile(f, dtype=np.float32)
#         data = data.reshape(-1, d + 1)[:, 1:].copy()  # Ignore the first element of each row (dimension of the vector)
#         return data


# def load_ivecs(file_path):
#     """加载 .ivecs 文件"""
#     with open(file_path, "rb") as f:
#         d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # 每个向量的长度（维度）
#         n = int(len(f.read()) / (d + 1) / 4)  # 计算向量数量
#         f.seek(0)  # 重置文件指针
#         data = np.fromfile(f, dtype=np.int32).reshape(-1, d + 1)[:, 1:]  # 忽略每行的第一个元素
#         return data


# def evaluate_index(index, queries, groundtruth, ef_search_list, k):
#     print(f"Index contains {index.get_current_count()} elements.")
#     """评估索引性能"""
#     recalls = []
#     qps_list = []
    
#     for ef_search in ef_search_list:
#         index.set_ef(ef_search)
        
#         # 测量搜索时间
#         start_time = time.time()
#         labels, _ = index.knn_query(queries, k=k)
#         elapsed_time = time.time() - start_time
        
#         # 计算 QPS
#         qps = len(queries) / elapsed_time
#         qps_list.append(qps)
        
#         # 计算 Recall
#         recall = np.mean([len(set(gt).intersection(set(l))) / k for gt, l in zip(groundtruth, labels)])
#         recalls.append(recall)
        
#         print(f"ef_search={ef_search}, QPS={qps:.2f}, Recall={recall:.4f}")
    
#     return recalls, qps_list


# def plot_recall_qps(recalls_marked, qps_marked, recalls_remaining, qps_remaining, ef_search_list, save_path):
#     """绘制 Recall-QPS 图并保存到指定路径"""
#     plt.figure(figsize=(8, 6))
    
#     # Marked index performance
#     plt.plot(recalls_marked, qps_marked, label="Marked Index", marker="o")
#     # Remaining index performance
#     plt.plot(recalls_remaining, qps_remaining, label="Remaining Index", marker="s")
    
#     plt.xlabel("Recall")
#     plt.ylabel("QPS (Queries Per Second)")
#     plt.title("Recall vs QPS")
#     plt.grid()
#     plt.legend()

#     # 保存图像
#     output_file = os.path.join(save_path, "recall_qps_comparison.png")
#     plt.savefig(output_file, format="png", dpi=300)
#     print(f"Recall-QPS graph saved at: {output_file}")

#     plt.show()


# def main():
#     # 文件路径
#     query_path = "/root/WorkSpace/dataset/sift/sift1M/sift_query.fvecs"
#     groundtruth_path = "/root/WorkSpace/dataset/sift/sift1M/output/groundtruth_remaining.ivecs"
#     mark_delete_ground_truth = "/root/WorkSpace/dataset/sift/sift1M/output/groundtruth_full_after_delete.ivecs"
#     marked_index_path = "/root/WorkSpace/dataset/sift/sift1M/output/hnsw_marked_index.bin"
#     remaining_index_path = "/root/WorkSpace/dataset/sift/sift1M/output/hnsw_remaining_index.bin"
#     output_dir = "/root/WorkSpace/dataset/sift/sift1M/output"  # 保存图像的路径
#     ef_search_list = [100,150, 200,250,300,350,400,450, 500]
#     k = 100  # Groundtruth的最近邻个数

#     # 加载查询数据和 groundtruth
#     print("Loading queries and groundtruth...")
#     queries = load_fvecs(query_path)
#     groundtruth = load_ivecs(groundtruth_path)
#     mark_delete_ground_truth = load_ivecs(mark_delete_ground_truth)

#     # 加载索引
#     print("Loading marked index...")
#     marked_index = hnswlib.Index(space="l2", dim=queries.shape[1])
#     marked_index.load_index(marked_index_path)
#     print("Loading remaining index...")
#     remaining_index = hnswlib.Index(space="l2", dim=queries.shape[1])
#     remaining_index.load_index(remaining_index_path)

#     # 评估 marked index
#     print("Evaluating marked index...")
#     recalls_marked, qps_marked = evaluate_index(marked_index, queries, mark_delete_ground_truth, ef_search_list, k)
    
#     # 评估 remaining index
#     print("Evaluating remaining index...")
#     recalls_remaining, qps_remaining = evaluate_index(remaining_index, queries, groundtruth, ef_search_list, k)

#     # 绘制 Recall-QPS 图并保存
#     print("Plotting and saving Recall-QPS graph...")
#     plot_recall_qps(recalls_marked, qps_marked, recalls_remaining, qps_remaining, ef_search_list, output_dir)


# if __name__ == "__main__":
#     main()


import hnswlib
import numpy as np
import os
import time
import matplotlib.pyplot as plt


def load_fvecs(filename):
    """Read .fvecs file format"""
    with open(filename, 'rb') as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # Dimensionality
        n = int(len(f.read()) / (d + 1) / 4)
        f.seek(0)
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(-1, d + 1)[:, 1:].copy()  # Ignore the first element of each row (dimension of the vector)
        return data


def load_ivecs(file_path):
    """加载 .ivecs 文件"""
    with open(file_path, "rb") as f:
        d = np.frombuffer(f.read(4), dtype=np.int32)[0]  # 每个向量的长度（维度）
        n = int(len(f.read()) / (d + 1) / 4)  # 计算向量数量
        f.seek(0)  # 重置文件指针
        data = np.fromfile(f, dtype=np.int32).reshape(-1, d + 1)[:, 1:]  # 忽略每行的第一个元素
        return data


def evaluate_index(index, queries, groundtruth, ef_search_list, k, num=3):
    """评估索引性能，重复 num 次并取平均值"""
    print(f"Index contains {index.get_current_count()} elements.")
    avg_recalls = []
    avg_qps_list = []

    for ef_search in ef_search_list:
        recalls = []
        qps_list = []

        for _ in range(num):
            index.set_ef(ef_search)

            # 测量搜索时间
            start_time = time.time()
            labels, _ = index.knn_query(queries, k=k)
            elapsed_time = time.time() - start_time

            # 计算 QPS
            qps = len(queries) / elapsed_time
            qps_list.append(qps)

            # 计算 Recall
            recall = np.mean([len(set(gt).intersection(set(l))) / k for gt, l in zip(groundtruth, labels)])
            recalls.append(recall)

        # 取多次运行的平均值
        avg_qps = np.mean(qps_list)
        avg_recall = np.mean(recalls)
        avg_qps_list.append(avg_qps)
        avg_recalls.append(avg_recall)

        print(f"ef_search={ef_search}, Avg QPS={avg_qps:.2f}, Avg Recall={avg_recall:.4f}")

    return avg_recalls, avg_qps_list


def plot_recall_qps(recalls_marked, qps_marked, recalls_remaining, qps_remaining, ef_search_list, save_path):
    """绘制 Recall-QPS 图并保存到指定路径"""
    plt.figure(figsize=(8, 6))
    
    # Marked index performance
    plt.plot(recalls_marked, qps_marked, label="Marked Index", marker="o")
    # Remaining index performance
    plt.plot(recalls_remaining, qps_remaining, label="Remaining Index", marker="s")
    
    plt.xlabel("Recall")
    plt.ylabel("QPS (Queries Per Second)")
    plt.title("Recall vs QPS")
    plt.grid()
    plt.legend()

    # 保存图像
    output_file = os.path.join(save_path, "recall_qps_comparison.png")
    plt.savefig(output_file, format="png", dpi=300)
    print(f"Recall-QPS graph saved at: {output_file}")

    plt.show()


def main():
    # 文件路径
    query_path = "/root/WorkSpace/dataset/sift/sift1M/sift_query.fvecs"
    groundtruth_path = "/root/WorkSpace/dataset/sift/sift1M/output/groundtruth_remaining.ivecs"
    mark_delete_ground_truth = "/root/WorkSpace/dataset/sift/sift1M/output/groundtruth_full_after_delete.ivecs"
    marked_index_path = "/root/WorkSpace/dataset/sift/sift1M/output/hnsw_marked_index.bin"
    remaining_index_path = "/root/WorkSpace/dataset/sift/sift1M/output/hnsw_remaining_index.bin"
    output_dir = "/root/WorkSpace/dataset/sift/sift1M/output"  # 保存图像的路径
    ef_search_list = [100, 150, 200, 250, 300, 350, 400, 450, 500]
    k = 100  # Groundtruth的最近邻个数
    num_repeats = 5  # 每次评估重复的次数

    # 加载查询数据和 groundtruth
    print("Loading queries and groundtruth...")
    queries = load_fvecs(query_path)
    groundtruth = load_ivecs(groundtruth_path)
    mark_delete_ground_truth = load_ivecs(mark_delete_ground_truth)

    # 加载索引
    print("Loading marked index...")
    marked_index = hnswlib.Index(space="l2", dim=queries.shape[1])
    marked_index.load_index(marked_index_path)
    print("Loading remaining index...")
    remaining_index = hnswlib.Index(space="l2", dim=queries.shape[1])
    remaining_index.load_index(remaining_index_path)

    # 评估 marked index
    print("Evaluating marked index...")
    recalls_marked, qps_marked = evaluate_index(marked_index, queries, mark_delete_ground_truth, ef_search_list, k, num=num_repeats)
    
    # 评估 remaining index
    print("Evaluating remaining index...")
    recalls_remaining, qps_remaining = evaluate_index(remaining_index, queries, groundtruth, ef_search_list, k, num=num_repeats)

    # 绘制 Recall-QPS 图并保存
    print("Plotting and saving Recall-QPS graph...")
    plot_recall_qps(recalls_marked, qps_marked, recalls_remaining, qps_remaining, ef_search_list, output_dir)


if __name__ == "__main__":
    main()
