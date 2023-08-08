### **about**

`mapletrees` is a header-only library for creating trees (currently only kdtrees) that may be useful for polygon meshing, with a focus on simplicity and efficiency for constructing and querying kdtrees (e.g. for k-nearest neighbors).

Currently, two types of trees are offered. The first tree is _balanced_ in that each node should have roughly the same number of nodes in the left and right subtrees. In this implementation, left and right children of a node are referenced by integers into the global array of nodes. The second tree is _left-balanced_ so that the number of nodes in the left subtree is always a power of two. In this case, the indices of the left and right children of a node with index `i` are `2 * i` and `2 * i + 1` (assuming the root node has index 1). This can help with cache coherence and can improve the efficiency of tree traversal (e.g. during queries).

Other trees useful for polygon meshing may be added in the future, such as octrees, bounding volume hierarchies, or alternating digital trees.

#### **why another kdtree library?**

I use kdtrees for computing Voronoi diagrams via the radius of security theorem, which requires the nearest neighbors to each Voronoi site. My goal was to reduce the memory footprint and try to improve the efficiency of the tree construction and query process. In developing the left-balanced kdtree implementation, my hope was that the querying procedures could eventually be ported to GPUs so that the nearest neighbor query and Voronoi cell construction could be done simultaneously in a kernel. I had tried writing the balanced-tree query process in CUDA but didn't see much of an improvement, likely because of memory coalescence and thread divergence. I haven't tried the query process yet with left-balanced trees (which is stackless, following Ingo Wald's recent work) on the GPU, but it might be more efficient. Please feel free to open a PR which implements this!

### **using the `maple` API**

All `mapletrees` functionality is enclosed within the `maple` namespace. `mapletrees` expects the input point coordinates to be stored in a single flattened array where the dimension of the points is the stride taken along the array when traversing from one point to the next. For example, `points[3 * k]` is the x-coordinate of the `k`-th point, while `points[3 * k + 1]` and `points[3 * k + 2]` are the y- and z-coordinates, respectively.

With this point representation, the balanced tree can be constructed as:

```c++
static constexpr int dim = 3;
const size_t n_points = 1e7; // 10M points
using coord_t = double; // the type of the point coordinates (either float or double)
using index_t = uint32_t; // how integers should be represented
std::vector<coord_t> points(n_points * 3); // ... and then fill point coordinates
maple::KdTreeOptions options; // it's optional to pass this to the constructor
maple::KdTree<dim, coord_t, index_t> tree(points.data(), n_points, options);
```

Alternatively, for a left-balanced tree, the last line would be:

```c++
static constexpr bool with_leaves = true;
maple::KdTree_LeftBalanced<dim, coord_t, index_t, with_leaves> tree(points.data(), n_points, options);
```

Note that for a left-balanced tree, there is an additional template parameter `with_leaves` which will build the tree with a bucket of points in the leaves if true. This is a template parameter because the internal structure used to represent nodes will require less memory if there are no leaf buckets, and construction will be faster. However, the query time will be slower if there are no buckets in the leaves. The number of points in a leaf bucket can be controlled by the `KdTreeOptions` structure (also applicable to regular balanced trees) via the `leaf_size` member.

#### nearest neighbor queries

Once the `tree` has been constructed, the k-nearest neighbors of an input query point `x` can be computed as in the following:

```c++
static constexpr int k = 10; // number of nearest neighbors requested
std::array<index_t, k> neighbors; // set up neighbors and distances buffers
std::array<coord_t, k> distances;
using Search_t = maple::NearestNeighborSearch<index_t, coord_t>;
Search_t search(k, neighbors.data(), distances.data());
tree.knearest(x, search);
```

The indices of the neighbors (and corresponding distances to `x`) will be filled in the input buffers `neighbors` and `distances`.

There is some initial support for a radius search (i.e. all points within a radius around an input point) but this is not well tested or optimized yet, so I don't recommend using it (another good idea for a PR!).

### **performance**

`mapletrees` was primarily compared with the `nanoflann` library since that is what I mostly used in the past. Overall, construction is faster than `nanoflann`, but querying can be slower. However, the crossover of when this happens usually occurs for problems that I don't generally work on, specifically when doing an all-nearest neighbor query for k > 100. Hopefully, the data included in the **Issues** tab will help to know which kdtree library to select for your application. The latest results are described [here](https://github.com/middpolymer/mapletrees/issues/1).

Both balanced and left-balanced (with leaf buckets) trees require `3 x sizeof(index_t)` bytes per point. The left-balanced tree without leaf buckets requires `sizeof(index_t)` bytes per point. `nanoflann` seems to require between 3-4 `x sizeof(index_t)` per point.

### **LICENSE**

`mapletrees` is distributed under the Apache-2.0 License.

Copyright 2023 Philip Claude Caplan

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
