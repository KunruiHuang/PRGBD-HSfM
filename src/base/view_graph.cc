#include "base/view_graph.h"

#include <glog/logging.h>

#include "base/database.h"
#include "util/connected_components.h"

namespace colmap {

ViewGraph::ViewGraph() = default;

int ViewGraph::NumImages() const { return static_cast<int>(vertices_.size()); }

int ViewGraph::NumEdges() const { return static_cast<int>(edges_.size()); }

bool ViewGraph::HasImage(const image_t image_id) const {
  return vertices_.find(image_id) != vertices_.end();
}

bool ViewGraph::HasEdge(const image_t image_id1,
                        const image_t image_id2) const {
  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  return edges_.find(pair_id) != edges_.end();
}

std::unordered_set<image_t> ViewGraph::ImageIds() const {
  std::unordered_set<image_t> image_ids;
  image_ids.reserve(vertices_.size());
  for (const auto& vertex : vertices_) {
    image_ids.insert(vertex.first);
  }

  return image_ids;
}

bool ViewGraph::RemoveImage(const image_t image_id) {
  const auto iter = vertices_.find(image_id);
  if (iter == vertices_.end()) return false;

  const auto& neighbor_ids = iter->second;

  for (const image_t neighbor_id : neighbor_ids) {
    vertices_[neighbor_id].erase(image_id);
    const image_pair_t image_pair_id =
        Database::ImagePairToPairId(image_id, neighbor_id);
    if (edges_.count(image_pair_id) > 0) {
      edges_.erase(image_pair_id);
    }
  }

  vertices_.erase(image_id);
  return true;
}

void ViewGraph::AddEdge(const image_t image_id1, const image_t image_id2,
                        const TwoViewGeometry& two_view_geometry) {
  if (image_id1 == kInvalidImageId || image_id2 == kInvalidImageId ||
      image_id1 == image_id2) {
    LOG(WARNING) << "Can not add and edge from " << image_id1 << " to "
                 << image_id2 << " itself ot invalid image id";
    return;
  }

  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  if (edges_.count(image_pair_id) > 0) {
    LOG(WARNING) << "An edge already exists between image " << image_id1
                 << " and image " << image_id2;
    return;
  }

  // 这是保存的是摄影测量的格式
  vertices_[image_id1].insert(image_id2);
  vertices_[image_id2].insert(image_id1);
  TwoViewGeometry two_view_info = two_view_geometry;
  if (Database::SwapImagePair(image_id1, image_id2)) {
    two_view_info.Invert();
  }

  edges_[image_pair_id].emplace_back(two_view_info);
}

bool ViewGraph::RemoveEdge(const image_t image_id1, const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  if (edges_.count(image_pair_id) == 0) {
    return false;
  }

  if (vertices_[image_id1].erase(image_id2) == 0 ||
      vertices_[image_id2].erase(image_id1) == 0 ||
      edges_.erase(image_pair_id) == 0) {
    // 这将会把所有的边全部的删除掉
    return false;
  }

  return true;
}

const std::unordered_set<image_t>* ViewGraph::GetNeighorIdsForImage(
    const image_t image_id) const {
  const auto iter = vertices_.find(image_id);
  if (iter == vertices_.end()) {
    LOG(WARNING) << "View graph does not have " << image_id;
    return nullptr;
  }

  return &iter->second;
}

const std::vector<TwoViewGeometry>* ViewGraph::GetEdge(const image_t image_id1,
                                          const image_t image_id2) const {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  const auto iter = edges_.find(image_pair_id);
  if (iter == edges_.end()) {
    LOG(WARNING) << "View graph does not have edge " << image_pair_id;
    return nullptr;
  }

  return &iter->second;
}

std::vector<TwoViewGeometry>* ViewGraph::GetMutableEdge(const image_t image_id1,
                                           const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  auto iter = edges_.find(image_pair_id);
  if (iter == edges_.end()) {
    LOG(WARNING) << "View graph does not have edge " << image_pair_id;
    return nullptr;
  }

  return &iter->second;
}

std::unordered_map<image_pair_t, std::vector<TwoViewGeometry>>& ViewGraph::GetAllEdges() {
  return edges_;
}

const std::unordered_map<image_pair_t, std::vector<TwoViewGeometry>>&
ViewGraph::GetAllEdges() const {
  return edges_;
}

void ViewGraph::ExtractSubgraph(
    const std::unordered_set<image_t>& views_in_subgraph,
    ViewGraph* sub_graph) const {
  CHECK_NOTNULL(sub_graph);

  for (const auto& vertex : vertices_) {
    if (views_in_subgraph.count(vertex.first) == 0) {
      continue;
    }

    for (const image_t& second_vertex : vertex.second) {
      if (views_in_subgraph.count(second_vertex) == 0 ||
          second_vertex < vertex.first) {
        continue;
      }

      const image_pair_t image_pair_id =
          Database::ImagePairToPairId(vertex.first, second_vertex);
      CHECK(HasEdge(vertex.first, second_vertex));
      const auto& edges = edges_.find(image_pair_id)->second;
      for (const auto& edge : edges) {
        sub_graph->AddEdge(vertex.first, second_vertex, edge);
      }
    }
  }
}

void ViewGraph::GetLargestConnectedComponentIds(
    std::unordered_set<image_t>* largest_cc) const {
  ConnectedComponents<image_t> cc_extractor;

  for (const auto& edge : edges_) {
    image_t image_id1;
    image_t image_id2;
    Database::PairIdToImagePair(edge.first, &image_id1, &image_id2);
    cc_extractor.AddEdge(image_id1, image_id2);
  }

  std::unordered_map<image_t, std::unordered_set<image_t>> connected_components;
  cc_extractor.Extract(&connected_components);

  // 遍历所有的连接成分, 找到连接成分中最大的那个
  image_t largest_cc_id = kInvalidImageId;
  size_t largest_cc_size = 0;
  for (const auto& connected_component : connected_components) {
    if (connected_component.second.size() > largest_cc_size) {
      largest_cc_size = connected_component.second.size();
      largest_cc_id = connected_component.first;
    }
  }

  CHECK_NE(largest_cc_id, kInvalidImageId);
  std::swap(*largest_cc, connected_components[largest_cc_id]);
}

}  // namespace colmap