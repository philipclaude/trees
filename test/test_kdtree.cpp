#include <fstream>

#include "kdtree.h"
#include "stlext.h"
#include "util.h"

struct json {
  json() { str = "{\n"; }
  template <typename T> std::string convert(const T& v) {
    return std::to_string(v);
  }
  template <typename T> void add(const std::string& key, const T& v) {
    if (n_keys > 0) str += ",\n";
    str += "\t\"" + key + "\": " + convert(v);
    n_keys++;
  }
  std::string dump() const { return str + "\n}\n"; }

  int n_keys{0};
  std::string str;
};

template <> std::string json::convert(const std::string& v) { return v; }

using index_t = uint32_t;
using coord_t = double;

int main(int argc, char** argv) {
  int64_t n = 1e3;
  const int dim = 3;
  Timer timer;
  bool use_zcurve = true;
  maple::NearestNeighborApproach approach =
      maple::NearestNeighborApproach::kRecursive;
  int n_neighbors = 10;
  int leaf_size = 12;
  int n_test = 1;
  int method = 0;  // 0 for balanced, 1 for left-balanced with leaves, 2 for
                   // leaft-balanced no leaves, 3 for nanoflann
  if (argc > 1) method = std::atoi(argv[1]);
  if (argc > 2) n = std::atoi(argv[2]);
  if (argc > 3) n_neighbors = std::atoi(argv[3]);
  if (argc > 4) leaf_size = std::atoi(argv[4]);
  if (argc > 5)
    approach = std::atoi(argv[5]) > 0
                   ? maple::NearestNeighborApproach::kIterative
                   : maple::NearestNeighborApproach::kRecursive;
  if (argc > 6) n_test = std::atoi(argv[6]);

  double t_zcurve = 0;
  double t_build = 0;
  double t_knn = 0;
  double memory = 0;
  int used_leaf_size = leaf_size;
  static const std::string method_name[] = {
      "balanced", "left-balanced-leaf-full", "left-balanced-leaf-empty",
      "nanoflann"};

  srand(0);
  for (int itest = 0; itest < n_test; itest++) {
    timer.start();
    std::vector<coord_t> coord(n * dim);
    std::vector<coord_t> points(n * dim);
    std::vector<index_t> order(n);
    timer.stop();
    std::cout << "alloc time = " << timer.seconds() << " s. " << std::endl;
    for (auto& x : coord) x = coord_t(rand()) / coord_t(RAND_MAX);

    if (use_zcurve) {
      timer.start();
      sort_points_on_zcurve(coord.data(), n, dim, order);
      timer.stop();
      std::cout << "zcurve time = " << timer.seconds() << " s. " << std::endl;
      t_zcurve += timer.seconds() / n_test;
    }

    std::parafor_i(0, n, [&](int thread_id, index_t i) {
      for (int d = 0; d < dim; d++)
        points[dim * i + d] = coord[dim * order[i] + d];
    });
    // free up memory that is no longer needed
    std::vector<coord_t>().swap(coord);
    std::vector<index_t>().swap(order);

    std::cout << "building kdtree for " << n / 1e6 << "M points" << std::endl;
    timer.start();
    maple::KdTreeOptions options;
    options.leaf_size = leaf_size;
    options.parallel = true;

    std::shared_ptr<maple::KdTreeNd<coord_t, index_t>> tree{nullptr};
    if (method == 0)
      tree = std::make_shared<maple::KdTree<dim, coord_t, index_t>>(
          points.data(), n, options);
    else if (method == 1)
      tree = std::make_shared<
          maple::KdTree_LeftBalanced<dim, coord_t, index_t, true>>(
          points.data(), n, options);
    else if (method == 2) {
      tree = std::make_shared<
          maple::KdTree_LeftBalanced<dim, coord_t, index_t, false>>(
          points.data(), n, options);
    } else if (method == 3) {
      tree = std::make_shared<KdTree_nanoflann<dim, coord_t, index_t>>(
          points.data(), n, options);
    }

    timer.stop();
    std::cout << "done in " << timer.seconds() << " s." << std::endl;
    t_build += timer.seconds() / n_test;
    if (n <= 1e2) tree->print();

    std::cout << "kdtree memory used = " << tree->gb() << "GB" << std::endl;
    memory += tree->gb() / n_test;

    std::cout << "testing " << n_neighbors << " nearest neighbors" << std::endl;
    timer.start();
    index_t n_thread = std::thread::hardware_concurrency();

    std::vector<int> n_k(n_thread, 0), n_p(n_thread);
    std::parafor_i(
        0, n,
        [&](int tid, size_t i) {
          int capacity = n_neighbors;
          index_t* neighbors = (index_t*)alloca(capacity * sizeof(index_t));
          coord_t* distances = (coord_t*)alloca(capacity * sizeof(coord_t));
          using Search_t = maple::NearestNeighborSearch<index_t, coord_t>;
          if (method == 3)
            approach = maple::NearestNeighborApproach::kRecursive;
          Search_t search(n_neighbors, neighbors, distances);
          search.approach = approach;

          tree->knearest(&points[i * dim], search);
          if (search.nearest() != i) {
            std::cout << "i = " << i << " neighbors 0 = " << search.nearest()
                      << std::endl;
            for (int i = 0; i < n_neighbors; i++) {
              std::cout << "j = " << search.neighbors[i]
                        << ", d = " << search.distances[i] << std::endl;
            }
          }
          assert(search.nearest() == i);
        },
        false);
    timer.stop();
    std::cout << "done in " << timer.seconds() << " s." << std::endl;
    t_knn += timer.seconds() / n_test;
    used_leaf_size = tree->leaf_size();
  }

  json stats;
  stats.add("method", method_name[method]);
  stats.add("n_samples", n_test);
  stats.add("n", n);
  stats.add("t_zcurve", t_zcurve);
  stats.add("t_build", t_build);
  stats.add("k", n_neighbors);
  stats.add("approach",
            approach == maple::NearestNeighborApproach::kRecursive ? 0 : 1);
  stats.add("t_knn", t_knn);
  stats.add("leaf_size", used_leaf_size);
  stats.add("memory", memory);

  std::string fname = "results-m" + std::to_string(method) + "-n" +
                      std::to_string(int(std::log10(n))) + "-k" +
                      std::to_string(n_neighbors) + "-l" +
                      std::to_string(leaf_size) + ".json";
  std::ofstream strm(fname);
  std::string s = stats.dump();
  std::cout << s << std::endl;
  strm << s << std::endl;
  strm.close();

  return 0;
}
