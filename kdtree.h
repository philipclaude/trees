//
// mapletrees
// Copyright 2023 Philip Claude Caplan
//
// Licensed under the Apache License,
// Version 2.0(the "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.
//
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <execution>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

#if HAVE_NANOFLANN
#include "nanoflann.hpp"
#endif

#ifdef NDEBUG
#define kdtree_assert(X)
#else
#define kdtree_assert(X) assert(X)
#endif

#define ASSERT(x)                                                        \
  {                                                                      \
    if (!(x)) {                                                          \
      std::cerr << "assertion failed: " << #x << " (" << __FILE__ << ":" \
                << __LINE__ << ")" << std::endl;                         \
      exit(1);                                                           \
    }                                                                    \
  }

namespace maple {

enum class NearestNeighborApproach : uint8_t { kRecursive = 0, kIterative = 1 };
struct KdTreeOptions {
  bool parallel{true};
  int leaf_size{16};
  int max_dim{-1};
};

template <typename index_t, typename coord_t>
struct NearestNeighborSearch;
template <typename index_t, typename coord_t>
struct RadiusSearch;

template <typename coord_t, typename index_t>
class KdTreeNd {
 public:
  virtual ~KdTreeNd() {}
  virtual void knearest(
      const coord_t* x,
      NearestNeighborSearch<index_t, coord_t>& search) const = 0;
  virtual void rsearch(const coord_t* x,
                       RadiusSearch<index_t, coord_t>& search) const = 0;
  virtual void print() const = 0;
  virtual double gb() const = 0;
  virtual int leaf_size() const = 0;
};

template <typename index_t, typename coord_t>
struct NearestNeighborSearch {
  int n_neighbors{0}, k{0};
  coord_t* distances{nullptr};
  index_t* neighbors{nullptr};
  NearestNeighborApproach approach{NearestNeighborApproach::kRecursive};

  NearestNeighborSearch(int n, index_t* n_buffer, coord_t* d_buffer)
      : k(n), neighbors(n_buffer), distances(d_buffer) {
    ASSERT(k > 1);
    reset();
  }

  inline void reset() {
    for (int i = 0; i < k; i++) {
      distances[i] = std::numeric_limits<coord_t>::max();
      neighbors[i] = std::numeric_limits<index_t>::max();
    }
    n_neighbors = 2;
  }

  inline void insert(index_t n, coord_t d) {
    int i = n_neighbors - 1;
    int j = i - 1;
    while (distances[j] > d && i > 0) {
      neighbors[i] = neighbors[j];
      distances[i] = distances[j];
      i = j;
      --j;
    }
    neighbors[i] = n;
    distances[i] = d;
    if (n_neighbors < k) ++n_neighbors;
  }

  inline index_t nearest() const { return neighbors[0]; }
  inline void finalize() {}
  inline coord_t max_distance() const { return distances[k - 1]; }
  inline int size() const { return n_neighbors; }
  inline bool full() const { return n_neighbors == k; }
};

template <typename index_t, typename coord_t>
struct RadiusSearch {
  int capacity{0}, n_neighbors{0}, k{0};
  coord_t* distances{nullptr};
  index_t* neighbors{nullptr};
  coord_t radius2{0.0};
  NearestNeighborApproach approach{NearestNeighborApproach::kRecursive};

  RadiusSearch(int n, index_t* n_buffer, coord_t* d_buffer)
      : k(n), neighbors(n_buffer), distances(d_buffer) {}

  inline void reset() {}

  inline void insert(index_t n, coord_t d) {
    if (n_neighbors >= capacity) return;
    assert(n_neighbors < capacity);
    neighbors[n_neighbors] = n;
    distances[n_neighbors] = d;
    n_neighbors++;
  }

  inline index_t nearest() const { return neighbors[0]; }

  void finalize() {
    using nn_pair = std::pair<coord_t, index_t>;
    nn_pair* data = (nn_pair*)alloca(n_neighbors * sizeof(nn_pair));
    for (size_t i = 0; i < n_neighbors; i++)
      data[i] = {distances[i], neighbors[i]};
    std::partial_sort(data, data + k, data + n_neighbors);
    for (size_t i = 0; i < n_neighbors; i++) {
      neighbors[i] = data[i].second;
      distances[i] = data[i].first;
    }
  }
  inline coord_t max_distance() const { return radius2; }
  inline int size() const { return n_neighbors; }
  inline bool full() const { return false; }
};

#ifndef __restrict__
#define __restrict__
#endif

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// internal definitions
static constexpr int8_t nullnode_ptr = 0;
static constexpr uint8_t n_axis_bits = 3;
static constexpr uint8_t n_aux_bits = 4;
static constexpr uint8_t max_3_bits = 7;
static constexpr uint8_t max_4_bits = 15;
static constexpr uint8_t leaf_axis_value = max_3_bits;

template <typename T>
T power_of_two(const T& x);

template <>
uint32_t power_of_two(const uint32_t& x) {
  return 0x80000000 >> __builtin_clz(x);
}

template <typename T, uint8_t dim>
inline T squared_distance(const T* __restrict__ x, const T* __restrict__ y) {
  T d = 0.0;
#pragma unroll
  for (uint8_t i = 0; i < dim; i++) {
    const T dx = x[i] - y[i];
    d += dx * dx;
  }
  return d;
}

static std::string node_type_name[3] = {"internal", "leaf", "bucket"};
template <typename index_t = uint32_t>
struct KdTreeNode {
  index_t index{nullnode_ptr};
  index_t left{nullnode_ptr};
  index_t right{nullnode_ptr};
  inline int8_t get_axis() const {
    return (index & 4) + (index & 2) + (index & 1);
  }
  void set_axis(int8_t axis) { index |= axis; }
  inline index_t get_index() const { return index >> n_axis_bits; }
  void set_index(index_t idx) {
    bool l = is_leaf();
    int8_t a = get_axis();
    index = idx << n_axis_bits;
    if (l) set_leaf();
    set_axis(a);
  }

  void set_leaf() { set_axis(max_3_bits); }
  inline bool is_leaf() const { return get_axis() == leaf_axis_value; }
  inline bool is_in_leaf() const {
    return get_axis() == leaf_axis_value && left == nullnode_ptr &&
           right == nullnode_ptr;
  }

  void print() const {
    int type = is_in_leaf() ? 2 : is_leaf() ? 1 : 0;
    int64_t l = is_in_leaf() ? -1 : is_leaf() ? left : int64_t(left) - 1;
    int64_t r = is_in_leaf() ? -1 : is_leaf() ? right : int64_t(right) - 1;
    int32_t a = is_in_leaf() ? -1 : int32_t(get_axis());
    std::cout << "node " << std::setw(4) << get_index() << " (" << std::setw(8)
              << node_type_name[type] << ") axis = " << std::setw(2) << a
              << " with children " << std::setw(4) << l << " and "
              << std::setw(4) << r << std::endl;
  }
};

template <int8_t dim, typename coord_t>
struct BoundingBox {
 public:
  BoundingBox() { reset(); }
  BoundingBox(const BoundingBox& b) {
    for (int d = 0; d < dim; ++d) {
      lo[d] = b.lo[d];
      hi[d] = b.hi[d];
    }
  }
  void reset() {
    for (int i = 0; i < dim; ++i) {
      lo[i] = std::numeric_limits<coord_t>::max();
      hi[i] = std::numeric_limits<coord_t>::min();
    }
  }

  void update(const coord_t* x) {
#pragma unroll
    for (int d = 0; d < dim; ++d)
      update(x[d], d);
  }

  void update(const coord_t& x, int d) {
    if (x < lo[d]) lo[d] = x;
    if (x > hi[d]) hi[d] = x;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const BoundingBox<dim, coord_t>& b) {
    os << "lo = (" << b.lo[0] << ", " << b.lo[1] << ", " << b.lo[2] << "), ";
    os << "hi = (" << b.hi[0] << ", " << b.hi[1] << ", " << b.hi[2] << ")";
    return os;
  }

  coord_t lo[dim], hi[dim];
};

template <int8_t dim, typename coord_t, typename index_t, bool _with_leaves>
class KdTree;
template <int8_t dim, typename coord_t, typename index_t, bool _with_leaves>
void build_level(KdTree<dim, coord_t, index_t, _with_leaves>* tree,
                 size_t idx_m,
                 size_t idx_l,
                 size_t idx_r,
                 int d,
                 int level) {
  tree->nodes()[idx_m].left = tree->build(idx_l, idx_m, d, ++level);
  tree->nodes()[idx_m].right = tree->build(idx_m + 1, idx_r, d, level);
}

template <int8_t dim,
          typename coord_t = double,
          typename index_t = uint32_t,
          bool _with_leaves = true>
class KdTree : public KdTreeNd<coord_t, index_t> {
 public:
  static_assert(dim <= 6, "dimension must be less than 7");
  using node_t = KdTreeNode<index_t>;
  using box_t = BoundingBox<dim, coord_t>;

  KdTree(const coord_t* points,
         index_t n_points,
         KdTreeOptions options = KdTreeOptions())
      : points_(points),
        nodes_(n_points),
        leaf_size_(options.leaf_size),
        max_dim_(options.max_dim) {
    assert(n_points << n_axis_bits < std::numeric_limits<index_t>::max());
    if (options.max_dim < 0) max_dim_ = dim;
    for (index_t i = 0; i < n_points; i++)
      nodes_[i].set_index(i);

    if (options.parallel) {
      size_t n_threads = std::thread::hardware_concurrency();
      threads_.reserve(n_threads);
      paralevel_ = std::ceil(std::log(n_threads) / std::log(2));
    }
    root_index_ = build(0, nodes_.size(), 0, 0);

    // calculate root bounding box
    for (size_t i = 0; i < n_points; i++)
      root_box_.update(points_ + i * dim);
  }

  index_t build(size_t idx_l, size_t idx_r, int axis, int32_t level) {
    // number of elements in this subtree
    const index_t n = idx_r - idx_l;
    if (n <= 0) return nullnode_ptr;
    size_t idx_m = idx_l + n / 2;

    // create a leaf if the number of nodes is less than the leaf size
    if (n <= leaf_size_) {
      for (auto j = idx_l; j < idx_r; j++) {
        nodes_[j].set_leaf();
        nodes_[j].left = idx_m + 1;  // nullnode_ptr;
        nodes_[j].right = nullnode_ptr;
      }
      nodes_[idx_m].set_leaf();
      nodes_[idx_m].left = idx_l;
      nodes_[idx_m].right = idx_r;

      return idx_m + 1;  // null nodes have index 0
    }

    // select the axis for the split
    coord_t xmin[dim], xmax[dim], xavg[dim];
#pragma unroll
    for (int d = 0; d < dim; d++) {
      xmin[d] = std::numeric_limits<coord_t>::max();
      xmax[d] = std::numeric_limits<coord_t>::min();
      xavg[d] = 0;
    }
    for (int j = idx_l; j < idx_r; ++j) {
      const coord_t* p = points_ + dim * nodes_[j].get_index();
#pragma unroll
      for (int d = 0; d < dim; ++d) {
        const coord_t& x = p[d];
        xavg[d] += x;
        if (x < xmin[d]) xmin[d] = x;
        if (x > xmax[d]) xmax[d] = x;
      }
    }

#pragma unroll
    for (int d = 0; d < dim; d++) {
      // xavg[d] /= n;
      // xavg[d] = 0.5 * (xmin[d] + xmax[d]);
      xavg[d] = points_[nodes_[idx_m].get_index() * dim + d];
    }

    axis = 0;
    coord_t lmax = xmax[0] - xmin[0];
#pragma unroll
    for (int d = 1; d < dim; ++d) {
      const coord_t ld = xmax[d] - xmin[d];
      if (ld > lmax) {
        axis = d;
        lmax = ld;
      }
    }

    // place all elements \lt or \gt the axis median value
    auto begin = nodes_.begin();
    std::nth_element(begin + idx_l, begin + idx_m, begin + idx_r,
                     [axis, this](const node_t& p, const node_t& q) {
                       return points_[dim * p.get_index() + axis] <
                              points_[dim * q.get_index() + axis];
                     });
    nodes_[idx_m].set_axis(axis);

    // build the children
    // axis = (axis + 1) % max_dim_;
    if (level != paralevel_) {
      nodes_[idx_m].left = build(idx_l, idx_m, axis, ++level);
      nodes_[idx_m].right = build(idx_m + 1, idx_r, axis, level);
    } else {
      threads_.push_back(
          std::thread(build_level<dim, coord_t, index_t, _with_leaves>, this,
                      idx_m, idx_l, idx_r, axis, level));
      if (threads_.size() == size_t(1 << paralevel_))
        for (auto& t : threads_)
          t.join();
    }

    return idx_m + 1;  // null nodes have index 0
  }

  index_t nearest(const coord_t* x,
                  NearestNeighborApproach approach =
                      NearestNeighborApproach::kRecursive) const {
    using Search_t = NearestNeighborSearch<index_t, coord_t>;
    index_t neighbors[2];
    coord_t distance[2];
    Search_t search{2, neighbors, distance};
    search.approach = approach;
    knearest(x, search);
    return search.nearest();
  }

  void knearest(const coord_t* x,
                NearestNeighborSearch<index_t, coord_t>& search) const {
    ASSERT(search.approach == NearestNeighborApproach::kRecursive);
    search_recursive(x, search);
    search.finalize();
  }

  void rsearch(const coord_t* x, RadiusSearch<index_t, coord_t>& search) const {
    ASSERT(search.approach == NearestNeighborApproach::kRecursive);
    search_recursive(x, search);
    search.finalize();
  }

  void print() const {
    std::cout << "root = " << root_index_ - 1 << std::endl;
    for (size_t i = 0; i < nodes_.size(); i++) {
      std::cout << "[" << i << "] ";
      nodes_[i].print();
    }
  }

  double gb() const {
    size_t n_bytes = nodes_.capacity() * sizeof(node_t);
    return double(n_bytes) / 1e9;
  }

  index_t max_level() const {
    index_t level = 0;
    const node_t* node = nodes_[root_index_ - 1];
    while (!node->is_leaf()) {
      if (node->left > 0)
        node = &nodes_[node->left - 1];
      else if (node->right > 0)
        node = &nodes_[node->right - 1];
      else
        break;
      level++;
    }
    return level;
  }

  auto& nodes() { return nodes_; }
  int leaf_size() const { return leaf_size_; }

 private:
  template <typename Search_t>
  inline void _search_recursive(const coord_t* __restrict__ x,
                                Search_t& result,
                                const node_t& node,
                                box_t& box,
                                coord_t box_dist) const {
    const index_t node_index = node.get_index();
    const index_t left = node.left;
    const index_t right = node.right;
    const coord_t* p = points_ + dim * node_index;
    const coord_t d = squared_distance<coord_t, dim>(x, p);
    const coord_t max_dist = result.max_distance();
    if (d < max_dist) result.insert(node_index, d);

    if (node.is_leaf()) {
      for (index_t j = left; j < right; j++) {
        const node_t& n = nodes_[j];
        const index_t leaf_index = n.get_index();
        const coord_t dn =
            squared_distance<coord_t, dim>(x, points_ + dim * leaf_index);
        if (dn < result.max_distance()) result.insert(leaf_index, dn);
      }
      return;
    }

    if (left == nullnode_ptr && right == nullnode_ptr) return;
    const int axis = node.get_axis();
    if (axis == leaf_axis_value) return;
    const coord_t p_axis = p[axis];
    const coord_t dx = x[axis] - p_axis;
    if (dx < 0) {
      coord_t box_d = box.hi[axis];
      box.hi[axis] = p_axis;
      if (likely(left > 0))
        _search_recursive(x, result, nodes_[left - 1], box, box_dist);
      box.hi[axis] = box_d;

      const coord_t dbox = box.lo[axis] - x[axis];
      if (dbox > 0) box_dist -= dbox * dbox;
      box_dist += dx * dx;

      if (result.max_distance() > box_dist) {
        box_d = box.lo[axis];
        box.lo[axis] = p_axis;
        if (likely(right > 0))
          _search_recursive(x, result, nodes_[right - 1], box, box_dist);
        box.lo[axis] = box_d;
      }
    } else {
      coord_t box_d = box.lo[axis];
      box.lo[axis] = p_axis;
      if (likely(right > 0))
        _search_recursive(x, result, nodes_[right - 1], box, box_dist);
      box.lo[axis] = box_d;

      const coord_t dbox = x[axis] - box.hi[axis];
      if (dbox > 0) box_dist -= dbox * dbox;
      box_dist += dx * dx;

      if (result.max_distance() > box_dist) {
        box_d = box.hi[axis];
        box.hi[axis] = p_axis;
        if (likely(left > 0))
          _search_recursive(x, result, nodes_[left - 1], box, box_dist);
        box.hi[axis] = box_d;
      }
    }
  }
  template <typename Search_t>
  void search_recursive(const coord_t* x, Search_t& result) const {
    box_t box = root_box_;
    coord_t box_dist = 0;
    for (int d = 0; d < dim; ++d) {
      if (x[d] < box.lo[d]) box_dist += (x[d] - box.lo[d]) * (x[d] - box.lo[d]);
      if (x[d] > box.hi[d]) box_dist += (x[d] - box.hi[d]) * (x[d] - box.hi[d]);
    }
    _search_recursive(x, result, nodes_[root_index_ - 1], box, box_dist);
  }

  const coord_t* points_;
  std::vector<node_t> nodes_;
  std::vector<std::thread> threads_;
  int paralevel_{-1};
  const int leaf_size_;
  index_t root_index_;
  int max_dim_;
  box_t root_box_;
};

template <int8_t dim, typename coord_t, typename index_t, bool with_leaves>
class KdTree_LeftBalanced;

template <int8_t dim, typename coord_t, typename index_t, bool with_leaves>
void build_level_left_balanced(
    KdTree_LeftBalanced<dim, coord_t, index_t, with_leaves>* tree,
    size_t nc,
    size_t nm,
    size_t nl,
    size_t nr,
    int d,
    int level) {
  if (nm - nl > 0) tree->build(nc << 1, nl, nm, d, level);
  if (nr - nm > 0) tree->build((nc << 1) + 1, nm + 1, nr, d, level);
}

namespace detail {
template <typename index_t, bool _with_leaves>
struct node_data_t;

template <typename index_t>
struct node_data_t<index_t, true> {
  node_data_t() : aux(0) {}
  index_t aux;
};

template <typename index_t>
struct node_data_t<index_t, false> {};

}  // namespace detail

template <int8_t dim,
          typename coord_t = double,
          typename index_t = uint32_t,
          bool with_leaves = false>
class KdTree_LeftBalanced : public KdTreeNd<coord_t, index_t> {
 public:
  static_assert(dim <= 6, "dimension must be less than 7");

  struct node_t : detail::node_data_t<index_t, with_leaves> {
    index_t index{0};
    inline int8_t get_axis() const {
      return (index & 4) + (index & 2) + (index & 1);
    }
    void set_axis(int8_t axis) { index |= axis; }
    inline index_t get_index() const { return index >> n_axis_bits; }
    void set_index(index_t idx) {
      bool l = is_leaf();
      int8_t a = get_axis();
      index = idx << n_axis_bits;
      if (l) set_leaf();
      set_axis(a);
    }

    inline int8_t get_n_leaf() const {
      if constexpr (with_leaves)
        return (this->aux & 8) + (this->aux & 4) + (this->aux & 2) +
               (this->aux & 1);
      else
        return 0;
    }
    void set_n_leaf(int8_t n) {
      if constexpr (with_leaves) this->aux |= n;
    }

    inline index_t get_aux() const {
      if constexpr (with_leaves) return this->aux >> n_aux_bits;
      return 0;
    }
    void set_aux(index_t val) {
      if constexpr (with_leaves) {
        int8_t n = get_n_leaf();
        this->aux = val << n_aux_bits;
        set_n_leaf(n);
      }
    }

    // a leaf should not use the axis, so it's okay to use 7 (max with 3
    // bits)
    void set_leaf() { set_axis(max_3_bits); }
    inline bool is_leaf() const { return get_axis() == leaf_axis_value; }
    inline bool is_in_leaf() const {
      if constexpr (with_leaves)
        return get_n_leaf() == 0 && get_axis() == leaf_axis_value;
      return false;
    }

    void print() const {
      int type = is_in_leaf() ? 2 : is_leaf() ? 1 : 0;
      int32_t a = is_in_leaf() ? -1 : int32_t(get_axis());
      std::cout << "node " << std::setw(4) << get_index() << " ("
                << std::setw(8) << node_type_name[type]
                << ") axis = " << std::setw(2) << a << std::endl;
    }
  };

  KdTree_LeftBalanced(const coord_t* points,
                      index_t n_points,
                      KdTreeOptions options = KdTreeOptions())
      : points_(points),
        nodes_(n_points + 1),
        index_(n_points),
        leaf_size_(std::min(options.leaf_size, int(max_4_bits))),
        max_dim_(options.max_dim) {
    const auto n_shift_bits = std::max(n_aux_bits, n_axis_bits);
    assert(n_points << n_shift_bits < std::numeric_limits<index_t>::max());
    if (options.max_dim < 0) max_dim_ = dim;
    for (index_t i = 0; i < n_points; i++)
      index_[i] = i;
    if constexpr (with_leaves) leaves_.resize(n_points);

    if (options.parallel && n_points > 64) {
      size_t n_threads = std::thread::hardware_concurrency();
      threads_.reserve(n_threads);
      paralevel_ = std::ceil(std::log(n_threads) / std::log(2));
    }
    build(1, 0, index_.size(), 0, 0);
    std::vector<index_t>().swap(index_);
    n_nodes_ = nodes_.size();
  }

  void build(size_t nc, size_t nl, size_t nr, int axis, int32_t level) {
    const index_t n = nr - nl;
    assert(n > 0);
    assert(nc < nodes_.size());

    if constexpr (with_leaves) {
      if (n <= leaf_size_) {
        lock_.lock();
        nodes_[nc].set_index(index_[nl]);
        nodes_[nc].set_leaf();
        nodes_[nc].set_n_leaf(n);
        nodes_[nc].set_aux(n_leaf_);
        for (int j = 0; j < n; j++)
          leaves_[n_leaf_ + j] = index_[nl + j];
        n_leaf_ += n;
        lock_.unlock();
        return;
      }
    } else {
      if (n == 1) {
        nodes_[nc].set_index(index_[nl]);
        nodes_[nc].set_leaf();
        return;
      }
    }

    // select the axis for the split
    coord_t xmin[dim], xmax[dim];
#pragma unroll
    for (int d = 0; d < dim; d++) {
      xmin[d] = std::numeric_limits<coord_t>::max();
      xmax[d] = std::numeric_limits<coord_t>::min();
    }
    for (int j = nl; j < nr; j++) {
      const coord_t* p = &points_[dim * index_[j]];
      for (int d = 0; d < dim; d++) {
        const coord_t& x = p[d];
        if (x < xmin[d]) xmin[d] = x;
        if (x > xmax[d]) xmax[d] = x;
      }
    }

    axis = 0;
    coord_t s = xmax[0] - xmin[0];
#pragma unroll
    for (int d = 1; d < dim; d++) {
      const coord_t ld = xmax[d] - xmin[d];
      if (ld > s) {
        axis = d;
        s = ld;
      }
    }

    // determine the highest power of two smaller than n
    // so the tree is left-balanced
    const index_t m = power_of_two(n);
    const index_t r = n - (m - 1);
    index_t lsize = (m - 2) / 2, rsize = lsize;
    if (r < m / 2) {
      lsize += r;
    } else {
      lsize += m / 2;
      rsize += r - m / 2;
    }

    // median index
    const size_t nm = nl + lsize;
    kdtree_assert(nm < index_.size());

    // place all elements \lt or \gt the axis median value
    auto begin = index_.begin();
    std::nth_element(begin + nl, begin + nm, begin + nr,
                     [axis, this](const index_t& p, const index_t& q) {
                       return points_[dim * p + axis] < points_[dim * q + axis];
                     });
    nodes_[nc].set_index(index_[nm]);
    nodes_[nc].set_axis(axis);

    // build the children
    if (level != paralevel_) {
      level++;
      if (lsize > 0) build(nc << 1, nl, nm, axis, level);
      if (rsize > 0) build((nc << 1) + 1, nm + 1, nr, axis, level);
    } else {
      threads_.push_back(std::thread(
          build_level_left_balanced<dim, coord_t, index_t, with_leaves>, this,
          nc, nm, nl, nr, axis, ++level));
      if (threads_.size() == size_t(1 << paralevel_))
        for (auto& t : threads_)
          t.join();
    }
  }

  index_t nearest(const coord_t* x,
                  NearestNeighborApproach approach =
                      NearestNeighborApproach::kRecursive) const {
    using Search_t = NearestNeighborSearch<index_t, coord_t>;
    index_t neighbors[2];
    coord_t distance[2];
    Search_t search{2, neighbors, distance};
    search.approach = approach;
    knearest(x, search);
    return search.neighbors[0];
  }

  void knearest(const coord_t* x,
                NearestNeighborSearch<index_t, coord_t>& search) const {
    if (search.approach == NearestNeighborApproach::kIterative)
      search_iterative(x, search);
    else {
      assert(search.approach == NearestNeighborApproach::kRecursive);
      search_recursive(x, search, 1);
    }
    search.finalize();
  }

  void rsearch(const coord_t* x, RadiusSearch<index_t, coord_t>& search) const {
    if (search.approach == NearestNeighborApproach::kIterative)
      search_iterative(x, search);
    else {
      assert(search.approach == NearestNeighborApproach::kRecursive);
      search_recursive(x, search, 1);
    }
    search.finalize();
  }

  void print() const {
    for (size_t i = 1; i < nodes_.size(); i++) {
      std::cout << "[" << i << "] ";
      nodes_[i].print();
    }
  }

  double gb() const {
    size_t n_bytes =
        nodes_.capacity() * sizeof(node_t) +
        index_.capacity() * sizeof(typename decltype(index_)::value_type) +
        leaves_.capacity() * sizeof(typename decltype(leaves_)::value_type);
    return double(n_bytes) / 1e9;
  }

  int leaf_size() const { return leaf_size_; }

 private:
  template <typename Search_t>
  inline void search_recursive(const coord_t* __restrict__ x,
                               Search_t& result,
                               const index_t idx) const {
    if (idx >= n_nodes_) return;
    const node_t& node = nodes_[idx];
    const index_t node_index = node.get_index();
    const coord_t* __restrict__ p = points_ + dim * node_index;
    const auto d = squared_distance<coord_t, dim>(x, p);
    const coord_t max_dist = result.max_distance();
    if (d < max_dist) result.insert(node_index, d);

    if constexpr (with_leaves) {
      const int8_t n_leaf = node.get_n_leaf();
      if (n_leaf > 0) {
        const index_t aux = node.get_aux();
        for (int j = 0; j < n_leaf; j++) {
          const index_t leaf_index = leaves_[aux + j];
          const coord_t* __restrict__ q = points_ + dim * leaf_index;
          const auto dl = squared_distance<coord_t, dim>(x, q);
          if (dl < result.max_distance()) result.insert(leaf_index, dl);
        }
        return;
      }
    }
    const int axis = node.get_axis();
    if (axis != leaf_axis_value) {
      // determine the next branch to visit
      const coord_t dx = p[axis] - x[axis];
      const index_t other = (idx << 1) - (dx > 0);
      const index_t next = other + 1;
      search_recursive(x, result, next);
      if (result.max_distance() > dx * dx) search_recursive(x, result, other);
    }
  }

  template <typename Search_t>
  void search_iterative(const coord_t* __restrict__ x, Search_t& result) const {
    // initialize search data
    coord_t max_dist = result.max_distance();
    index_t prev = 0;
    index_t curr = 1;
    index_t next = 1;

    while (next) {
      const index_t parent = curr >> 1;
      if (curr == n_nodes_) {
        prev = curr;
        curr = parent;
        continue;
      }
      const uint64_t child = curr << 1;

      const bool from_child = !(prev < child);
      const auto& node = nodes_[curr];
      const index_t node_index = node.get_index();
      const coord_t* p = points_ + dim * node_index;
      if (!from_child) {
        // only add this node on the way down the tree
        const auto d = squared_distance<coord_t, dim>(x, p);
        if (d < max_dist) {
          result.insert(node_index, d);
          max_dist = result.max_distance();
        }
      }

      const int axis = node.get_axis();
      // check if we are at a leaf node
      if (axis == leaf_axis_value) {
        if constexpr (with_leaves) {
          const int8_t n_leaf = node.get_n_leaf();
          for (int j = 0; j < n_leaf; ++j) {
            const index_t leaf_index = leaves_[node.get_aux() + j];
            const coord_t* q = &points_[dim * leaf_index];
            const auto dl = squared_distance<coord_t, dim>(x, q);
            if (dl < max_dist) {
              result.insert(leaf_index, dl);
              max_dist = result.max_distance();
            }
          }
        }
        prev = curr;
        curr = parent;
        continue;
      }
      const coord_t dx = p[axis] - x[axis];
      index_t best = (curr << 1);
      index_t other = best + 1;
      if (dx < 0) {
        ++best;
        --other;
      }

      next = parent;
      if (prev == best && other < n_nodes_) {
        if (dx * dx < max_dist) next = other;
      } else if (prev != other && child < n_nodes_)
        next = best;

      prev = curr;
      curr = next;
    }
  }

  index_t n_nodes_;
  const coord_t* points_;
  std::vector<node_t> nodes_;
  std::vector<index_t> index_;
  std::vector<index_t> leaves_;
  size_t n_leaf_{0};
  std::vector<std::thread> threads_;
  int paralevel_{-1};
  const int leaf_size_;
  int max_dim_;
  std::mutex lock_;
};

}  // namespace maple

template <int8_t dim,
          typename coord_t = double,
          typename index_t = uint32_t,
          bool with_leaves = true>
class KdTree_nanoflann : public maple::KdTreeNd<coord_t, index_t> {
 private:
  class PointCloud {
   public:
    PointCloud(const coord_t* points, size_t n_points)
        : points_(points), n_points_(n_points) {}

    size_t kdtree_get_point_count() const { return n_points_; }
    coord_t kdtree_get_pt(const size_t k, int d) const {
      return points_[dim * k + d];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
      return false;
    }

   private:
    const coord_t* points_;
    size_t n_points_;
  };

 public:
  KdTree_nanoflann(const coord_t* x,
                   size_t n,
                   maple::KdTreeOptions options = maple::KdTreeOptions())
      : cloud_(x, n) {
    tree_ = std::make_unique<nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Adaptor<coord_t, PointCloud>, PointCloud, dim> >(
        dim, cloud_,
        nanoflann::KDTreeSingleIndexAdaptorParams(
            options.leaf_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None,
            options.parallel ? std::thread::hardware_concurrency() : 1));
  }

  void knearest(const coord_t* x,
                maple::NearestNeighborSearch<index_t, coord_t>& search) const {
    (void)tree_->knnSearch(x, search.k, search.neighbors, search.distances);
  }

  void rsearch(const coord_t* x,
               maple::RadiusSearch<index_t, coord_t>& search) const {
    std::cout << "not implemented" << std::endl;
    ASSERT(false);
  }

  index_t nearest(const coord_t* x) const {
    coord_t distance;
    index_t neighbor;
    tree_->knnSearch(x, 1, &neighbor, &distance);
    return neighbor;
  }

  double gb() const { return double(tree_->usedMemory(*tree_)) / 1e9; }
  void print() const {}
  int leaf_size() const { return tree_->leaf_max_size_; }

 private:
  PointCloud cloud_;
  std::unique_ptr<nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Adaptor<coord_t, PointCloud>,
      PointCloud,
      dim> >
      tree_;
};

// #endif
