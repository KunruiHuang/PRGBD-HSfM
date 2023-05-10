#ifndef COLMAP_VIEW_GRAPH_H
#define COLMAP_VIEW_GRAPH_H

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "base/image.h"
#include "estimators/two_view_geometry.h"
#include "util/types.h"

namespace colmap {

// ViewGraph用于维护当前重建过程中所有的顶点和边之间的关系
// 注意两个顶点之间允许多条边存在
class ViewGraph {
 public:
  ViewGraph();

  int NumImages() const;

  int NumEdges() const;

  bool HasImage(const image_t image_id) const;

  bool HasEdge(const image_t image_id1, const image_t image_id2) const;

  std::unordered_set<image_t> ImageIds() const;

  bool RemoveImage(const image_t image_id);

  void AddEdge(const image_t image_id1, const image_t image_id2,
               const TwoViewGeometry& two_view_geometry);

  bool RemoveEdge(const image_t image_id1, const image_t image_id2);

  const std::unordered_set<image_t>* GetNeighorIdsForImage(
      const image_t image_id) const;

  const std::vector<TwoViewGeometry>* GetEdge(const image_t image_id1,
                                 const image_t image_id2) const;

  std::vector<TwoViewGeometry>* GetMutableEdge(const image_t image_id1,
                                  const image_t image_id2);

  std::unordered_map<image_pair_t, std::vector<TwoViewGeometry>>& GetAllEdges();
  const std::unordered_map<image_pair_t, std::vector<TwoViewGeometry>>& GetAllEdges() const;

  void ExtractSubgraph(const std::unordered_set<image_t>& views_in_subgraph,
                       ViewGraph* sub_graph) const;

  void GetLargestConnectedComponentIds(
      std::unordered_set<image_t>* largest_cc) const;

 private:
  std::unordered_map<image_t, std::unordered_set<image_t>> vertices_;
  std::unordered_map<image_pair_t, std::vector<TwoViewGeometry>> edges_;
};

}  // namespace colmap

#endif  // COLMAP_VIEW_GRAPH_H
