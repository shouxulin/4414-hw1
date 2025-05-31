## Goal
Get familar with the concept of *k*-d trees.

## Background

The *k*-d tree (short for *k*-dimensional tree) is a space-partitioning data structure used to organize points in a *k*-dimensional space. It is particularly useful for tasks that involve multidimensional search, such as nearest neighbor queries. A *k*-d tree is a binary tree where each node represents a point in *k*-dimensional space and implicitly defines a splitting hyperplane that partitions the space into two half-spaces. The splitting dimension cycles through the *k* coordinates at each level of the tree. One common application is **k-nearest neighbors (KNN) search**, where the goal is to efficiently find the *k* closest points in the dataset to a given query point. By leveraging the spatial structure of the *k*-d tree, KNN queries can prune large portions of the search space and reduce the number of distance computations. For more details, refer to the [Wikipedia article on k-d trees](https://en.wikipedia.org/wiki/K-d_tree).


## Code Overview

In the starter code, we have provided two key components: `Embedding_T` and `Node`. These are template structures that serve as the foundation for building and querying a K-D tree.

### `Embedding_T`:
The `Embedding_T` structure defines how to interpret and compute distances between data points in different embedding types. It uses C++ template specialization to support both 1-dimensional (`float`) and N-dimensional (`std::array<float, N>`) data:

- For scalar `float`, the distance function computes the absolute difference.
- For `std::array<float, N>`, the distance function computes the standard Euclidean distance.

Each specialization also provides a static constant `Dim` that indicates the dimensionality of the data.

This abstraction allows the K-D tree implementation to work with both scalar and vector embeddings in a generic and extensible way.

#### `Node<T>`

The `Node<T>` structure represents a node in the K-D tree. Each node contains:

- `embedding`: the point in *k*-dimensional space (of type `T`)
- `idx`: the index of the point in the dataset
- `left` and `right`: pointers to the left and right child nodes
- `queryEmbedding`: a static member used during nearest neighbor search to compare against a fixed query point

This structure enables recursive construction and traversal of the K-D tree, and will be the building block for implementing the tree construction and KNN search.


## Compile and Run

To **compile** the program, run:

```bash
g++ knn.cpp -o knn
```

To run the program:

```bash
./knn <mode> <input_data> <K>
```
- `<mode>`:  
  - Use `0` for scalar input (`float`)  
  - Use any other value for vector input (`std::array<float, N>`)

- `<input_data>`: Path to the input JSON data file

- `<K>`: Number of nearest neighbors to find

Example:
```bash
./knn 0 ./data/sample_data.json 3
```


## Your job

In the starter code, we have already extracted all data points from the input file. The **first** data point is treated as the **query point**, and the **remaining points** are used to build the K-D tree.

Your tasks are as follows:

- **Build a balanced K-D tree** using the dataset (excluding the first point).  
  For performance reasons, your tree should be balanced. One common strategy is:
  - At each level of the tree, select the **median** point (with respect to the splitting axis) to insert into the subtree.
  - The splitting axis cycles through the available dimensions (e.g., 0, 1, ..., *k*-1).

- **Implement K-nearest neighbor (KNN) search** on the K-D tree.  
  Given the query point (the first entry), your algorithm should return the **K closest points** in the dataset.

- **Write the results to a JSON file**:
  - The output file should be named:
    - `"neighbors-scalar"` if `mode = 0`
    - `"neighbors-vector"` otherwise
  - Each element in the JSON file should be a dictionary with the following keys:
    1. `"id"`: the index of the point in the original input file  
    2. `"dist"`: the distance between this neighbor and the query point  
    3. `"feature"`: the actual scalar or vector data  
    4. `"text"`: the corresponding text (i.e., string representation of the scalar or the text that the vector represents)
---

### Part 1: Scalar Points

In this part, each data point is a **single scalar value** (i.e., a float).  
Run the program with `<mode> = 0`.

We generate **100 random floats**, and the data is stored in: data/data_1d.json.


### Part 2: Vector Points

In this part, each data point is a **vector** (e.g., `std::array<float, N>`).  
Run the program with `<mode> = 1`.

We use the [MS MARCO dataset](https://microsoft.github.io/msmarco/), a large-scale benchmark created by Microsoft for evaluating machine reading comprehension and ranking tasks. Each example in this dataset includes real user queries and relevant documents.

Each vector represents an **embedding** of a short text sequence (e.g., a sentence or phrase). These embeddings are precomputed using a neural model and capture semantic meaning: similar texts will have vectors that are close in the embedding space.  
The goal is to use KNN search on these vector representations to find semantically similar text entries.

The original data is stored in ./data/OpenKPEval.tar. To unzip it:
```bash
tar -xzf ./data/OpenKPEval.tar
```

We provided a preprocessing scipt to convert the original data into a json file, use the following command to run it:
```bash
python preprocess.py
```

By default, this will extract 1000 vectors.








## Part 3
TBD