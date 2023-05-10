#ifndef COLMAP_CONNECTED_COMPONENTS_H
#define COLMAP_CONNECTED_COMPONENTS_H

#include <cstdint>
#include <glog/logging.h>

#include <limits>
#include <unordered_map>
#include <unordered_set>

template <typename T>
class ConnectedComponents {
 public:
  struct Root {
    Root(const T& id, const int size) : id(id), size(size) {}
    T id;
    int size;
  };

  ConnectedComponents()
      : max_connected_component_size_(std::numeric_limits<T>::max()) {}

  explicit ConnectedComponents(const int max_size)
      : max_connected_component_size_(max_size) {
    CHECK_GT(max_connected_component_size_, 0);
  }

  void AddEdge(const T& node1, const T& node2) {
    Root* root1 = FindOrInsert(node1);
    Root* root2 = FindOrInsert(node2);

    if (root1->id == root2->id ||
        (uint64_t)root1->size + (uint64_t)root2->size > max_connected_component_size_) {
      return;
    }

    if (root1->size < root2->size) {
      root2->size += root1->size;
      *root1 = *root2;
    } else {
      root1->size += root2->size;
      *root2 = *root1;
    }
  }

  void Extract(
      std::unordered_map<T, std::unordered_set<T> >* connected_components) {
    CHECK_NOTNULL(connected_components)->clear();

    for (const auto& node : disjoint_set_) {
      const Root* root = FindRoot(node.first);
      (*connected_components)[root->id].insert(node.first);
    }
  }

  bool NodesInSameConnectedComponent(const T& node1, const T& node2) {
    if (disjoint_set_.count(node1) == 0 || disjoint_set_.count(node2) == 0) {
      return false;
    }

    const Root* root1 = FindRoot(node1);
    const Root* root2 = FindRoot(node2);
    return root1->id == root2->id;
  }

 private:
  Root* FindOrInsert(const T& node) {
    const auto iter = disjoint_set_.find(node);
    const Root* parent = iter == disjoint_set_.end() ? nullptr : &iter->second;

    if (parent == nullptr) {
      disjoint_set_.insert(std::make_pair(node, Root(node, 1)));
      const auto iter2 = disjoint_set_.find(node);
      return &iter2->second;
    }

    return FindRoot(node);
  }

  Root* FindRoot(const T& node) {
    auto iter = disjoint_set_.find(node);
    Root* parent = &iter->second;

    // If this node is a root, return the node itself.
    if (node == parent->id) {
      return parent;
    }

    // Otherwise, recusively search for the root.
    Root* root = FindRoot(parent->id);
    *parent = *root;
    return root;
  }

  uint64_t max_connected_component_size_;

  std::unordered_map<T, Root> disjoint_set_;
};

#endif  // COLMAP_CONNECTED_COMPONENTS_H
