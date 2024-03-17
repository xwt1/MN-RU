//
// Created by root on 3/10/24.
//

#include "file/file.h"


template<class T>
void WriteOpt::WriteCSVFile(const std::vector<std::vector<T> >& data,const std::string & filePath) {
//    std::vector <std::vector<std::string>> coordinate={{"X","Y","Z"}}; //3D
    std::vector <std::vector<std::string>> coordinate={{"X","Y"}}; //2D
    std::ofstream outfile(filePath);
    if (outfile.is_open()) { // 检查文件是否成功打开
        for(const auto& row : coordinate){
            for(auto record: row){
                outfile << record<<",";
            }
            outfile<<"\n";
        }
        for(const auto& row : data){
            for(auto record: row){
                outfile << record<<",";
            }
            outfile<<"\n";
        }
        outfile.close(); // 关闭文件
        std::cout << "data successfully write in:"<<filePath<<".\n";
    } else {
        std::cout << "Unable to open file.\n";
    }
}

//void WriteOpt::WriteCSVFile(const std::vector<std::vector<float> >& data,const std::string & filePath) {
//    std::ofstream outfile(filePath);
//    if (outfile.is_open()) { // 检查文件是否成功打开
//        for(const auto& row : data){
//            for(auto record: row){
//                outfile << record<<",";
//            }
//            outfile<<"\n";
//        }
//        outfile.close(); // 关闭文件
//        std::cout << "data successfully write in:"<<filePath<<".\n";
//    } else {
//        std::cout << "Unable to open file.\n";
//    }
//}

void WriteOpt::WriteTXTFile(std::vector<std::string> &data, const std::string &filePath) {
    std::ofstream outfile(filePath);
    if (outfile.is_open()) { // 检查文件是否成功打开
        for(auto row : data){
            outfile << row << "\n";
        }
        outfile.close(); // 关闭文件
//        std::cout << "data successfully write in:"<<filePath<<".\n";
    } else {
        std::cout << "Unable to open file.\n";
    }
}

std::vector<std::vector<float>> WriteOpt::ConvertTo2DVector(float* data, int dim, int max_elements) {
    std::vector<std::vector<float>> result;
    result.reserve(max_elements); // 预分配空间以提高效率

    for (int i = 0; i < max_elements; ++i) {
        std::vector<float> element;
        element.reserve(dim); // 预分配空间以提高效率

        for (int j = 0; j < dim; ++j) {
            element.push_back(data[i * dim + j]);
        }

        result.push_back(element);
    }

    return result;
}

// 显式实例化
template void WriteOpt::WriteCSVFile<float>(const std::vector<std::vector<float>>& data, const std::string& filePath);
//
//template void WriteOpt::WriteCSVFile<std::string>(const std::vector<std::vector<std::string>>& data, const std::string& filePath);