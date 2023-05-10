#ifndef COLMAP_MINIMUM_SPANNING_TREE_H
#define COLMAP_MINIMUM_SPANNING_TREE_H

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "util/connected_components.h"
#include "util/hash.h"
#include "util/util.h"

namespace colmap {

template <typename T, typename V>
class MinimumSpanningTree {
 public:
  MinimumSpanningTree() {}

  // Add an edge in the graph.
  void AddEdge(const T& node1, const T& node2, const V& weight) {
    edges_.emplace_back(weight, std::pair<T, T>(node1, node2));
  }

  // Extracts the minimum spanning tree. Returns true on success and false upon
  // failure. If true is returned, the output variable contains the edge list of
  // the minimum spanning tree.
  bool Extract(std::unordered_set<std::pair<T, T> >* minimum_spanning_tree) {
    if (edges_.empty()) {
      std::cerr << "No edges were passed to the minimum spanning tree extractor!" << std::endl;
      return false;
    }

    // Determine the number of nodes in the graph.
    const int num_nodes = CountNodesInGraph();

    // Reserve space in the MST since we know it will have exactly N - 1 edges.
    minimum_spanning_tree->reserve(num_nodes - 1);

    // Order all edges by their weights.
    std::sort(edges_.begin(), edges_.end());

    // For each edge in the graph, add it to the minimum spanning tree if it
    // does not create a cycle.
    ConnectedComponents<T> cc;
    for (size_t i = 0;
         i < edges_.size() && minimum_spanning_tree->size() < (size_t)num_nodes - 1;
         i++) {
      const auto& edge = edges_[i];
      if (!cc.NodesInSameConnectedComponent(edge.second.first,
                                            edge.second.second)) {
        cc.AddEdge(edge.second.first, edge.second.second);
        minimum_spanning_tree->emplace(edge.second.first, edge.second.second);
      }
    }

    return minimum_spanning_tree->size() == static_cast<size_t>(num_nodes - 1);
  }

 private:
  // Counts the number of nodes in the graph by counting the number of unique
  // node values we have received from AddEdge.
  int CountNodesInGraph() {
    std::vector<T> nodes;
    nodes.reserve(edges_.size() * 2);
    for (const auto& edge : edges_) {
      nodes.emplace_back(edge.second.first);
      nodes.emplace_back(edge.second.second);
    }
    std::sort(nodes.begin(), nodes.end());
    auto unique_end = std::unique(nodes.begin(), nodes.end());
    return std::distance(nodes.begin(), unique_end);
  }

  std::vector<std::pair<V, std::pair<T, T> > > edges_;

  // Each node is mapped to a Root node. If the node is equal to the root id
  // then the node is a root and the size of the root is the size of the
  // connected component.
  std::unordered_map<T, T> disjoint_set_;

  DISALLOW_COPY_AND_ASSIGN(MinimumSpanningTree);
};

} // colmap

#endif  // COLMAP_MINIMUM_SPANNING_TREE_H
