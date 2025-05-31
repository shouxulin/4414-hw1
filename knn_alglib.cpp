/*
g++ -std=c++11 -I ~/course/hw1/alglib-cpp/src knn_arglib.cpp ~/course/hw1/alglib-cpp/src/*.cpp -o knn_arglib

./knn_arglib ./data/data_short.json
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "alglibmisc.h"
#include <nlohmann/json.hpp>


using json = nlohmann::json;


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <data.json> <K> <eps>\n";
        return 1;
    }

    // Load and parse JSON file
    std::ifstream ifs(argv[1]);
    if (!ifs) {
        std::cerr << "Error opening file: " << argv[1] << "\n";
        return 1;
    }
    json j;
    ifs >> j;
    if (!j.is_array() || j.size() < 2) {
        std::cerr << "JSON must be an array of at least 2 elements\n";
        return 1;
    }

    // Convert JSON array to a dict mapping id -> element
    std::unordered_map<int, json> dict;
    for (auto &elem : j) {
        int id = elem["id"].get<int>();
        dict[id] = elem;
    }

    try{

        auto query_obj   = j[0];
        size_t D         = query_obj["feature"].size();
        alglib::real_1d_array query;
        query.setlength(D);
        for (size_t d = 0; d < D; ++d) {
            query[d] = query_obj["feature"][d].get<double>();
        }

        size_t N_points = j.size() - 1;
        // Flatten dataset features into a contiguous array
        std::vector<double> raw((size_t)N_points * D);
        alglib::integer_1d_array tags;
        tags.setlength(N_points);

        for (size_t i = 0; i < N_points; ++i) {
            auto &elem = j[i+1];
            // fill features
            for (size_t d = 0; d < D; ++d) {
                raw[i*D + d] = elem["feature"][d].get<double>();
            }
            // fill tag
            tags[i] = elem["id"].get<int>();
        }

        // Wrap in alglib arrays
        alglib::real_2d_array points;
        points.setcontent((int)N_points, (int)D, raw.data());

        // Build the tagged k-d tree
        alglib::kdtree tree;
        // last two ints: normtype=0 (Euclidean), node capacity=2
        alglib::kdtreebuildtagged(points, tags, (int)N_points, (int)D, 0, 2, tree);

        // Perform approximate K-NN search
        int k = std::stoi(argv[2]);
        double eps = std::atof(argv[3]);
        alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);

        // Retrieve results
        alglib::integer_1d_array   idx;
        alglib::real_1d_array      dist;
        idx.setlength(count);
        dist.setlength(count);
        alglib::kdtreequeryresultstags(tree,    idx);
        alglib::kdtreequeryresultsdistances(tree, dist);

        std::cout << "query:\n";
        std::cout << "  feature: " << query_obj["feature"] << "\n";
        std::cout << "  text:    " << query_obj["text"] << "\n\n";


        for (int i = 0; i < count; ++i) {
            int tag = idx[i];
            auto &elem = dict[tag];
            auto &feat = elem["feature"];
            std::string text = elem["text"].get<std::string>();

            std::cout << "Neighbor " << i+1 << ":\n";
            std::cout << "  id: " << tag << ", dist= " << dist[i] << "\n";

            std::cout << "  feature: [";
            for (size_t d = 0; d < feat.size(); ++d) {
                std::cout << feat[d].get<double>()
                        << (d+1 < feat.size() ? ", " : "");
            }
            std::cout << "]\n";

            std::cout << "  text: \"" << text << "\"\n";
        }
    }
    catch(alglib::ap_error &e) {
        std::cerr << "ALGLIB error: " << e.msg << std::endl;
        return 1;
    }

    return 0;
}