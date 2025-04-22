# import hnswlib
# import numpy as np
# import os
# import random
# from sklearn.neighbors import NearestNeighbors


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


# def save_groundtruth_ivecs(data, query, k, save_path):
#     """计算 groundtruth 并保存为 .ivecs 格式，仅保存最近邻索引"""
#     nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
#     _, indices = nn.kneighbors(query)

#     with open(save_path, "wb") as f:
#         for row in indices:
#             row = np.array([len(row)] + list(row), dtype=np.int32)
#             row.tofile(f)


# def save_full_groundtruth(data, query, k, remaining_indices, save_path):
#     """
#     计算全集数据集删除点后的 groundtruth，但索引仍然保留全集数据集中的位置
#     """
#     nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
#     _, indices = nn.kneighbors(query)

#     # 将 indices 转换为全集索引位置
#     full_indices = np.vectorize(lambda x: remaining_indices[x])(indices)

#     with open(save_path, "wb") as f:
#         for row in full_indices:
#             row = np.array([len(row)] + list(row), dtype=np.int32)
#             row.tofile(f)


# def main(dataset_path, queryset_path, output_dir, k=100):
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)

#     # 加载主数据集和查询数据
#     print("Loading dataset and query data...")
#     dataset = load_fvecs(dataset_path)
#     queryset = load_fvecs(queryset_path)
#     index_dim = dataset.shape[1]

#     # Step 1: 构建全集 HNSW 索引并保存
#     print("Building full HNSW index...")
#     hnsw_index = hnswlib.Index(space="l2", dim=index_dim)
#     hnsw_index.init_index(max_elements=dataset.shape[0], ef_construction=200, M=16)
#     hnsw_index.add_items(dataset)

#     # Step 2: 随机标记前 50% 的点为删除
#     total_points = dataset.shape[0]
#     print(f"total_points {total_points} elements.")
#     delete_indices = random.sample(range(total_points), total_points // 2)
#     remaining_indices = sorted(set(range(total_points)) - set(delete_indices))
#     for idx in delete_indices:
#         hnsw_index.mark_deleted(idx)


#     print(f"remaining_indices {len(remaining_indices)} elements.")

#     # 保存标记删除后的索引
#     marked_index_path = os.path.join(output_dir, "hnsw_marked_index.bin")
#     hnsw_index.save_index(marked_index_path)
#     print(f"Marked HNSW index saved at: {marked_index_path}")

#     # 保存剩余点的索引
#     remaining_indices_path = os.path.join(output_dir, "remaining_indices.ivecs")
#     with open(remaining_indices_path, "wb") as f:
#         for idx in remaining_indices:
#             row = np.array([1, idx], dtype=np.int32)
#             row.tofile(f)
#     print(f"Remaining indices saved at: {remaining_indices_path}")

#     # Step 3: 使用剩余点构建 HNSW 索引并保存
#     remaining_data = dataset[remaining_indices]
#     print("Building remaining HNSW index...")
#     hnsw_remaining_index = hnswlib.Index(space="l2", dim=index_dim)
#     hnsw_remaining_index.init_index(max_elements=len(remaining_data), ef_construction=200, M=16)
#     hnsw_remaining_index.add_items(remaining_data)
#     hnsw_remaining_index_path = os.path.join(output_dir, "hnsw_remaining_index.bin")
#     hnsw_remaining_index.save_index(hnsw_remaining_index_path)
#     print(f"Remaining HNSW index saved at: {hnsw_remaining_index_path}")

#     # Step 4: 使用剩余点构建 groundtruth 并保存为 .ivecs
#     remaining_groundtruth_path_ivecs = os.path.join(output_dir, "groundtruth_remaining.ivecs")
#     save_groundtruth_ivecs(remaining_data, queryset, k, remaining_groundtruth_path_ivecs)
#     print(f"Remaining groundtruth saved at: {remaining_groundtruth_path_ivecs}")

#     # Step 5: 构建全集数据集删除点后的 groundtruth，并保持全集索引
#     full_groundtruth_path_ivecs = os.path.join(output_dir, "groundtruth_full_after_delete.ivecs")
#     save_full_groundtruth(remaining_data, queryset, k, remaining_indices, full_groundtruth_path_ivecs)
#     print(f"Full groundtruth after delete saved at: {full_groundtruth_path_ivecs}")


# if __name__ == "__main__":
#     # 替换以下路径为实际的路径
#     dataset_path = "/root/WorkSpace/dataset/sift/sift1M/sift_base.fvecs"
#     queryset_path = "/root/WorkSpace/dataset/sift/sift1M/sift_query.fvecs"
#     output_dir = "/root/WorkSpace/dataset/sift/sift1M/output"
#     k = 100

#     main(dataset_path, queryset_path, output_dir, k)



import hnswlib
import numpy as np
import os
import random
from sklearn.neighbors import NearestNeighbors


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


def save_groundtruth_ivecs(data, query, k, save_path):
    """计算 groundtruth 并保存为 .ivecs 格式，仅保存最近邻索引"""
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    _, indices = nn.kneighbors(query)

    with open(save_path, "wb") as f:
        for row in indices:
            row = np.array([len(row)] + list(row), dtype=np.int32)
            row.tofile(f)


def save_full_groundtruth(data, query, k, remaining_indices, save_path):
    """
    计算全集数据集删除点后的 groundtruth，但索引仍然保留全集数据集中的位置
    """
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(data)
    _, indices = nn.kneighbors(query)

    # 将 indices 转换为全集索引位置
    full_indices = np.vectorize(lambda x: remaining_indices[x])(indices)

    with open(save_path, "wb") as f:
        for row in full_indices:
            row = np.array([len(row)] + list(row), dtype=np.int32)
            row.tofile(f)


def main(dataset_path, queryset_path, output_dir, k=100, mark_delete_ratio=0.5):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载主数据集和查询数据
    print("Loading dataset and query data...")
    dataset = load_fvecs(dataset_path)
    queryset = load_fvecs(queryset_path)
    index_dim = dataset.shape[1]

    # Step 1: 构建全集 HNSW 索引并保存
    print("Building full HNSW index...")
    hnsw_index = hnswlib.Index(space="l2", dim=index_dim)
    hnsw_index.init_index(max_elements=dataset.shape[0], ef_construction=200, M=16)
    hnsw_index.add_items(dataset)

    # Step 2: 随机标记点为删除
    total_points = dataset.shape[0]
    delete_count = int(total_points * mark_delete_ratio)
    print(f"Marking {delete_count} points as deleted (total points: {total_points}).")
    delete_indices = random.sample(range(total_points), delete_count)
    remaining_indices = sorted(set(range(total_points)) - set(delete_indices))
    for idx in delete_indices:
        hnsw_index.mark_deleted(idx)

    print(f"remaining_indices {len(remaining_indices)} elements.")

    # 保存标记删除后的索引
    marked_index_path = os.path.join(output_dir, "hnsw_marked_index.bin")
    hnsw_index.save_index(marked_index_path)
    print(f"Marked HNSW index saved at: {marked_index_path}")

    # 保存剩余点的索引
    remaining_indices_path = os.path.join(output_dir, "remaining_indices.ivecs")
    with open(remaining_indices_path, "wb") as f:
        for idx in remaining_indices:
            row = np.array([1, idx], dtype=np.int32)
            row.tofile(f)
    print(f"Remaining indices saved at: {remaining_indices_path}")

    # Step 3: 使用剩余点构建 HNSW 索引并保存
    remaining_data = dataset[remaining_indices]
    print("Building remaining HNSW index...")
    hnsw_remaining_index = hnswlib.Index(space="l2", dim=index_dim)
    hnsw_remaining_index.init_index(max_elements=len(remaining_data), ef_construction=200, M=16)
    hnsw_remaining_index.add_items(remaining_data)
    hnsw_remaining_index_path = os.path.join(output_dir, "hnsw_remaining_index.bin")
    hnsw_remaining_index.save_index(hnsw_remaining_index_path)
    print(f"Remaining HNSW index saved at: {hnsw_remaining_index_path}")

    # Step 4: 使用剩余点构建 groundtruth 并保存为 .ivecs
    remaining_groundtruth_path_ivecs = os.path.join(output_dir, "groundtruth_remaining.ivecs")
    save_groundtruth_ivecs(remaining_data, queryset, k, remaining_groundtruth_path_ivecs)
    print(f"Remaining groundtruth saved at: {remaining_groundtruth_path_ivecs}")

    # Step 5: 构建全集数据集删除点后的 groundtruth，并保持全集索引
    full_groundtruth_path_ivecs = os.path.join(output_dir, "groundtruth_full_after_delete.ivecs")
    save_full_groundtruth(remaining_data, queryset, k, remaining_indices, full_groundtruth_path_ivecs)
    print(f"Full groundtruth after delete saved at: {full_groundtruth_path_ivecs}")


if __name__ == "__main__":
    # 替换以下路径为实际的路径
    dataset_path = "/root/WorkSpace/dataset/sift/sift1M/sift_base.fvecs"
    queryset_path = "/root/WorkSpace/dataset/sift/sift1M/sift_query.fvecs"
    output_dir = "/root/WorkSpace/dataset/sift/sift1M/output"
    k = 100
    mark_delete_ratio = 0.3  # 设置标记删除的比例

    main(dataset_path, queryset_path, output_dir, k, mark_delete_ratio)
