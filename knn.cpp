#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <queue>
#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static constexpr size_t Dim = 1;

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
};

// fixed-size array: N-D
template <size_t N>
struct Embedding_T<std::array<float, N>>
{
    static constexpr size_t Dim = N;

    static float distance(const std::array<float, N> &a,
                          const std::array<float, N> &b)
    {
        float s = 0;
        for (size_t i = 0; i < N; ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};

// extract the “axis”-th coordinate or the scalar itself
template<typename T>
constexpr float getCoordinate(T const &e, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return e;          // scalar case
    } else {
        return e[axis];    // vector case
    }
}

// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

template <typename T>
T Node<T>::queryEmbedding;


// Build a balanced KD‐tree by splitting on median at each level.
template <typename T>
Node<T>* buildKD(std::vector<std::pair<T,int>>& items, int depth = 0) {
    if (items.empty()) return nullptr;

    constexpr size_t D = Embedding_T<T>::Dim;
    size_t axis = depth % D;

    // Comparator that compares along the chosen axis:
    auto cmpAxis = [axis](auto const &a, auto const &b) {
        return getCoordinate(a.first, axis)
             < getCoordinate(b.first, axis);
    };

    // Find median position:
    size_t mid = items.size() / 2;
    std::nth_element(items.begin(), items.begin() + mid, items.end(), cmpAxis);

    // The median element becomes the root of this subtree:
    auto        medianPair = items[mid];
    T           medianEmb  = medianPair.first; 
    int         medianId   = medianPair.second;

    // Create a new node with (embedding, idx):
    Node<T>* node = new Node<T>{medianEmb, medianId};

    // Build the left half [0 .. mid-1]:
    if (mid > 0) {
        std::vector<std::pair<T,int>> leftItems(items.begin(),
                                               items.begin() + mid);
        node->left = buildKD(leftItems, depth + 1);
    } else {
        node->left = nullptr;
    }

    // Build the right half [mid+1 .. end]:
    if (mid + 1 < items.size()) {
        std::vector<std::pair<T,int>> rightItems(items.begin() + mid + 1,
                                                items.end());
        node->right = buildKD(rightItems, depth + 1);
    } else {
        node->right = nullptr;
    }

    return node;
}


// K-NN search with pruning
template <typename T>
using PQItem = std::pair<float, int>;
template <typename T>
using MaxHeap = std::priority_queue<
    PQItem<T>,
    std::vector<PQItem<T>>,
    std::less<PQItem<T>>>;

template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap<T> &heap)
{
    if (!node)
        return;

    // consider this node
    float dist = Embedding_T<T>::distance(
        node->embedding,
        Node<T>::queryEmbedding);
    if ((int)heap.size() < K)
    {
        heap.push({dist, node->idx});
    }
    else if (dist < heap.top().first)
    {
        heap.pop();
        heap.push({dist, node->idx});
    }

    // choose near/far child
    constexpr size_t D = Embedding_T<T>::Dim;
    size_t axis = depth % D;
    bool goLeft = [](auto const &q, auto const &e, size_t ax)
    {
        if constexpr (std::is_same_v<T, float>)
            return q < e;
        else
            return q[ax] < e[ax];
    }(Node<T>::queryEmbedding, node->embedding, axis);

    Node<T> *near = goLeft ? node->left : node->right;
    Node<T> *far = goLeft ? node->right : node->left;

    // explore near
    knnSearch(near, depth + 1, K, heap);

    // check if we need to explore far
    float worstDist = heap.empty() ? std::numeric_limits<float>::infinity()
                                   : heap.top().first;
    float delta = [](auto const &q, auto const &e, size_t ax)
    {
        if constexpr (std::is_same_v<T, float>)
            return std::abs(q - e);
        else
            return std::abs(q[ax] - e[ax]);
    }(Node<T>::queryEmbedding, node->embedding, axis);

    if ((int)heap.size() < K || delta < worstDist)
    {
        knnSearch(far, depth + 1, K, heap);
    }
}


template <typename T>
int runMain(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <data.json>\n";
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

    // Convert JSON array to dict id->object (used later for lookup)
    std::unordered_map<int, json> dict;
    for (auto &elem : j) {
        int id = elem["id"].get<int>();
        dict[id] = elem;
    }

    // select K NNs
    int K = std::stoi(argv[2]);

    // Extract the query embedding from j[0]
    auto query_obj = j[0];
    T    qemb;
    if constexpr (std::is_same_v<T, float>) {
        qemb = query_obj["feature"].get<float>();
    } else {
        for (size_t i = 0; i < Embedding_T<T>::Dim; ++i) {
            qemb[i] = query_obj["feature"][i].get<float>();
        }
    }
    Node<T>::queryEmbedding = qemb;


    // Collect all remaining points (IDs and embeddings) into a vector
    std::vector<std::pair<T,int>> allPoints;
    allPoints.reserve(j.size() - 1);
    for (size_t i = 1; i < j.size(); ++i) {
        auto &elem = j[i];
        T     emb;

        if constexpr (std::is_same_v<T, float>) {
            emb = elem["feature"].get<float>();
        } else {
            for (size_t k = 0; k < Embedding_T<T>::Dim; ++k) {
                emb[k] = elem["feature"][k].get<float>();
            }
        }

        int idx = elem["id"].get<int>();
        allPoints.emplace_back(emb, idx);
    }


    // Build balanced KD‐tree
    Node<T>* root = buildKD(allPoints, 0);

    // Perform K‐NN search and collect results
    MaxHeap<T> heap;
    knnSearch(root, 0, K, heap);

    // Collect and sort ascending by distance
    std::vector<PQItem<T>> out;
    while (!heap.empty()) {
        out.push_back(heap.top());
        heap.pop();
    }
    std::sort(out.begin(), out.end(),
              [](auto &a, auto &b) { return a.first < b.first; });

    // Print query and its top‐K neighbors
    std::cout << "query:\n";
    std::cout << "  feature: " << query_obj["feature"] << "\n";
    std::cout << "  text:    " << query_obj["text"] << "\n\n";

    nlohmann::json output_json = nlohmann::json::array();

    for (int i = 0; i < (int)out.size(); ++i) {
        auto &p      = out[i];
        float dist   = p.first;
        int   idx    = p.second;
        auto &elem   = dict[idx];

        std::cout << "Neighbor " << (i + 1) << ":\n";
        std::cout << "  id:      " << idx
                  << ", dist = " << dist << "\n";
        std::cout << "  feature: " << elem["feature"] << "\n";
        std::cout << "  text:    " << elem["text"] << "\n\n";

        nlohmann::json entry;
        entry["id"]      = idx;
        entry["dist"]    = dist;
        entry["feature"] = elem["feature"];
        entry["text"]    = elem["text"];

        output_json.push_back(entry);
    }


    std::string output_json_file = (std::is_same_v<T, float>) ? "neighbors_scalar.json" : "neighbors_vector.json";
    std::ofstream file(output_json_file);
    file << output_json.dump(2);
    file.close();

    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <mode> <data.json> <K>\n";
        return 1;
    }

    // mode 0: scalar float, mode 1: fixed-size array<float,20>
    int mode = std::stoi(argv[1]);


    char* new_argv[3];
    new_argv[0] = argv[0];   // keep original program name
    new_argv[1] = argv[2];   // pass JSON‐filename as argv[1]
    new_argv[2] = argv[3];   // pass K as argv[2]

    if (mode == 0) {
        return runMain<float>(2, new_argv);
    } else {
        return runMain<std::array<float,20>>(2, new_argv);
    }
}
