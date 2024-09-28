//
// Created by root on 6/6/24.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <exception>
#include <mutex>
#include "hnswlib/hnswlib.h"
#include "util.h"

int main() {

    std::string root_path = "/home/xiaowentao/WorkSpace/training-plan/dockerimages/delete_update/retrieval-diversity-enhancement";

    std::string data_path = "/root/WorkSpace/dataset/sift/sift200M/sift200M.fvecs";
    std::string index_path = "/root/WorkSpace/dataset/sift/sift200M/index/sift_200M_index.bin";

    int dim, num_points;
    std::vector<std::vector<float>> data = util::load_fvecs(data_path, dim, num_points);



    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> appr_alg(&space, num_points, 16, 500);

//    num_points =500000;

//    for(int i = 0 ;i < data.size() ; i ++){
//        appr_alg.checkTotalInOutDegreeEquality();
//        appr_alg.addPoint(data[i].data(),i);
//    }

    size_t numThreads = std::thread::hardware_concurrency();
    util::ParallelFor(0, num_points, numThreads, [&](size_t id, size_t threadId) {
        appr_alg.addPoint(data[id].data(), id);
    });

//    appr_alg.checkTotalInOutDegreeEquality();

    appr_alg.saveIndex(index_path);

    std::cout << "HNSW index created and saved to " << index_path << std::endl;

//    hnswlib::HierarchicalNSW<float> appr_alg2(&space, index_path, false, num_points, true);

//    std::cout<<"ok"<<std::endl;
    return 0;
}