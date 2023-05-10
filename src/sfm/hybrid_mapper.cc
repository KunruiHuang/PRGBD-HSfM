#include "sfm/hybrid_mapper.h"

#include <Eigen/Core>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <glog/logging.h>

#include <array>
#include <fstream>
#include <memory>

#include "base/cost_functions.h"
#include "base/projection.h"
#include "base/triangulation.h"
#include "estimators/pose.h"
#include "rgbd/rgbd.h"
#include "sfm/global/robust_rotation_estimator.h"
#include "util/bitmap.h"
#include "util/connected_components.h"
#include "util/minimum_spanning_tree.h"
#include "util/misc.h"

namespace colmap {
namespace {

using HeapElement = std::pair<TwoViewGeometry, std::pair<image_t, image_t>>;

bool SortHeapElement(const HeapElement& h1, const HeapElement& h2) {
  return h1.first.inlier_matches.size() > h2.first.inlier_matches.size();
}

Eigen::Vector3d ComputeOrientation(const Eigen::Vector3d& source_orientation,
                                   const TwoViewGeometry& two_view_info,
                                   const image_t source_view_id,
                                   const image_t neighbor_view_id) {
  Eigen::Matrix3d source_rotation_mat, relative_rotation;
  ceres::AngleAxisToRotationMatrix(
      source_orientation.data(),
      ceres::ColumnMajorAdapter3x3(source_rotation_mat.data()));
  relative_rotation = QuaternionToRotationMatrix(two_view_info.qvec);

  // get Rc2w
  const Eigen::Matrix3d neighbor_orientation =
      (source_view_id < neighbor_view_id)
          ? (relative_rotation * source_rotation_mat).eval()
          : (relative_rotation.transpose() * source_rotation_mat).eval();

  Eigen::Vector3d orientation;
  ceres::RotationMatrixToAngleAxis(
      ceres::ColumnMajorAdapter3x3(neighbor_orientation.data()),
      orientation.data());
  return orientation;
}

void AddEdgesToHeap(
    const ViewGraph& view_graph,
    const std::unordered_map<image_t, Eigen::Vector3d>& orientations,
    const image_t view_id, std::vector<HeapElement>* heap) {
  const auto* edge_ids = view_graph.GetNeighorIdsForImage(view_id);
  for (const image_t edge_id : *edge_ids) {
    if (ContainsKey(orientations, edge_id)) {
      continue;
    }

    // add two view geometry
    const auto* edges = view_graph.GetEdge(view_id, edge_id);
    for (const auto& edge : *edges) {
      heap->emplace_back(edge, std::make_pair(view_id, edge_id));
    }

    std::push_heap(heap->begin(), heap->end(), SortHeapElement);
  }
}

bool AngularDifferenceIsAcceptable(const Eigen::Vector3d& orientation1,
                                   const Eigen::Vector3d& orientation2,
                                   const Eigen::Vector3d& relative_orientation,
                                   const double sq_max_rot_change_rad) {
  const Eigen::Vector3d composed_relative_rotation =
      MultiplyRotations(orientation2, -orientation1);
  const Eigen::Vector3d loop_rotation =
      MultiplyRotations(-relative_orientation, composed_relative_rotation);
  const double rot_change_rad = loop_rotation.squaredNorm();
  return rot_change_rad <= sq_max_rot_change_rad;
}

void SortAndAppendNextImages(std::vector<std::pair<image_t, float>> image_ranks,
                             std::vector<image_t>* sorted_images_ids) {
  std::sort(image_ranks.begin(), image_ranks.end(),
            [](const std::pair<image_t, float>& image1,
               const std::pair<image_t, float>& image2) {
              return image1.second > image2.second;
            });

  sorted_images_ids->reserve(sorted_images_ids->size() + image_ranks.size());
  for (const auto& image : image_ranks) {
    sorted_images_ids->push_back(image.first);
  }

  image_ranks.clear();
}

float RankNextImageMaxVisiblePointsNum(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D());
}

float RankNextImageMaxVisiblePointsRatio(const Image& image) {
  return static_cast<float>(image.NumVisiblePoints3D()) /
         static_cast<float>(image.NumObservations());
}

float RankNextImageMinUncertainty(const Image& image) {
  return static_cast<float>(image.Point3DVisibilityScore());
}

}  // namespace

bool HybridMapper::Options::Check() const {
  CHECK_OPTION_GT(init_min_num_inliers, 0);
  CHECK_OPTION_GT(init_max_error, 0.0);
  CHECK_OPTION_GE(init_max_forward_motion, 0.0);
  CHECK_OPTION_LE(init_max_forward_motion, 1.0);
  CHECK_OPTION_GE(init_min_tri_angle, 0.0);
  CHECK_OPTION_GE(init_max_reg_trials, 1);
  CHECK_OPTION_GT(abs_pose_max_error, 0.0);
  CHECK_OPTION_GT(abs_pose_min_num_inliers, 0);
  CHECK_OPTION_GE(abs_pose_min_inlier_ratio, 0.0);
  CHECK_OPTION_LE(abs_pose_min_inlier_ratio, 1.0);
  CHECK_OPTION_GE(local_ba_num_images, 2);
  CHECK_OPTION_GE(local_ba_min_tri_angle, 0.0);
  CHECK_OPTION_GE(min_focal_length_ratio, 0.0);
  CHECK_OPTION_GE(max_focal_length_ratio, min_focal_length_ratio);
  CHECK_OPTION_GE(max_extra_param, 0.0);
  CHECK_OPTION_GE(filter_max_reproj_error, 0.0);
  CHECK_OPTION_GE(filter_min_tri_angle, 0.0);
  CHECK_OPTION_GE(max_reg_trials, 1);
  return true;
}

HybridMapper::HybridMapper(DatabaseCache* database_cache)
    : database_cache_(database_cache),
      reconstruction_(nullptr),
      triangulator_(nullptr),
      num_total_reg_images_(0),
      num_shared_reg_images_(0),
      prev_init_image_pair_id_(kInvalidImagePairId) {}

void HybridMapper::BeginReconstruction(Reconstruction* reconstruction) {
  CHECK(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  reconstruction_->SetUp(&database_cache_->CorrespondenceGraph(),
                         &database_cache_->ViewGraph());
  triangulator_ = std::make_unique<IncrementalTriangulator>(
      &database_cache_->CorrespondenceGraph(), reconstruction);

  num_shared_reg_images_ = 0;
  num_reg_images_per_camera_.clear();
  for (const image_t image_id : reconstruction_->RegImageIds()) {
    RegisterImageEvent(image_id);
  }

  existing_image_ids_ =
      std::unordered_set<image_t>(reconstruction->RegImageIds().begin(),
                                  reconstruction->RegImageIds().end());

  prev_init_image_pair_id_ = kInvalidImagePairId;
  prev_init_two_view_geometry_ = TwoViewGeometry();

  filtered_images_.clear();
  num_reg_trials_.clear();
}

void HybridMapper::SetUpReconstruction(Reconstruction* reconstruction) {
  reconstruction_ = reconstruction;
}

void HybridMapper::EndReconstruction(const bool discard) {
  CHECK_NOTNULL(reconstruction_);

  if (discard) {
    for (const image_t image_id : reconstruction_->RegImageIds()) {
      DeRegisterImageEvent(image_id);
    }
  }

  reconstruction_->TearDown();
  reconstruction_ = nullptr;
  triangulator_.reset();
}

bool HybridMapper::FilterInitialViewGraph(const int min_num_two_view_inliers) {
  const size_t min_inliers = static_cast<size_t>(min_num_two_view_inliers);
  std::unordered_set<image_pair_t> view_pairs_to_remove;
  const auto& view_pairs = reconstruction_->GetViewGraph()->GetAllEdges();
  const int before_num_view_pairs = static_cast<int>(view_pairs.size());
  for (const auto& view_pair : view_pairs) {
    for (const auto& edge : view_pair.second) {
      if (edge.config == TwoViewGeometry::ARKIT ||
          edge.config == TwoViewGeometry::RGBD) {
        continue;
      }

      if (edge.inlier_matches.size() < min_inliers) {
        view_pairs_to_remove.insert(view_pair.first);
      }
    }
  }

  for (const image_pair_t view_id_pair : view_pairs_to_remove) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(view_id_pair, &image_id1, &image_id2);
  }

  RemoveDisconnectedView(reconstruction_->GetViewGraph());

  std::cout << "  => Removed view pairs: "
            << before_num_view_pairs -
                   reconstruction_->GetViewGraph()->NumEdges()
            << std::endl;

  return reconstruction_->GetViewGraph()->NumEdges() >= 1;
}

bool HybridMapper::EstimateGlobalRotations() {
  PrintHeading2("Estimate Global Rotation");
  const auto& view_pairs = reconstruction_->GetViewGraph()->GetAllEdges();
  const auto* view_graph = reconstruction_->GetViewGraph();
  if (view_graph->NumEdges() == 0) {
    std::cerr << "View graph does not have edges." << std::endl;
    return false;
  }
  std::cout << "  => View graph node: " << view_graph->NumImages() << std::endl;
  std::cout << "  => View graph edge: " << view_graph->NumEdges() << std::endl;
  std::cout << "  => Initialize orientations from maximum spanning tree."
            << std::endl;
  OrientationsFromMaximumSpanningTree(*view_graph,
                                      &reconstruction_->GlobalOrientations());

  RobustRotationEstimator::Options options;
  std::unique_ptr<RobustRotationEstimator> rotation_estimator =
      std::make_unique<RobustRotationEstimator>(options);

  if (!rotation_estimator->EstimateRotations(
          view_pairs, &reconstruction_->GlobalOrientations())) {
    std::cerr << "Robust global rotations solver faliled!";
    return false;
  }

  std::cout << "  => Finish estimate global rotations." << std::endl;

  return true;
}

void HybridMapper::FilterRotations(
    const double max_relative_rotation_difference_degrees) {
  FilterViewPairsFromOrientation(max_relative_rotation_difference_degrees);

  const std::unordered_set<image_t> removed_views =
      RemoveDisconnectedView(reconstruction_->GetViewGraph());
  for (const auto removed_view : removed_views) {
    reconstruction_->GlobalOrientations().erase(removed_view);
  }

  std::cout << "  => Filter view: " << removed_views.size() << std::endl;
}

bool HybridMapper::FindInitialImagePair(const Options& options,
                                        image_t* image_id1,
                                        image_t* image_id2) {
  std::vector<image_t> image_ids1;
  if (*image_id1 != kInvalidImageId && *image_id2 == kInvalidImageId) {
    // Only *image_id1 provided.
    if (!database_cache_->ExistsImage(*image_id1)) {
      return false;
    }
    image_ids1.push_back(*image_id1);
  } else if (*image_id1 == kInvalidImageId && *image_id2 != kInvalidImageId) {
    // Only *image_id2 provided.
    if (!database_cache_->ExistsImage(*image_id2)) {
      return false;
    }
    image_ids1.push_back(*image_id2);
  } else {
    // No initial seed image provided
    image_ids1 = FindFirstInitialImage(options);
  }

  // Try to find good initial pair.
  for (size_t i1 = 0; i1 < image_ids1.size(); ++i1) {
    *image_id1 = image_ids1[i1];

    const std::vector<image_t> image_ids2 =
        FindSecondInitialImage(options, *image_id1);

    for (size_t i2 = 0; i2 < image_ids2.size(); ++i2) {
      *image_id2 = image_ids2[i2];

      const image_pair_t pair_id =
          Database::ImagePairToPairId(*image_id1, *image_id2);

      // Try every pair only once.
      if (init_image_pairs_.count(pair_id) > 0) {
        continue;
      }

      init_image_pairs_.insert(pair_id);

      if (EstimateInitialTwoViewGeometry(options, *image_id1, *image_id2)) {
        return true;
      }
    }
  }

  // No suitable pair found in entire dataset.
  *image_id1 = kInvalidImageId;
  *image_id2 = kInvalidImageId;

  return false;
}

std::vector<image_t> HybridMapper::FindNextImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  std::function<float(const Image&)> rank_image_func;
  switch (options.image_selection_method) {
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_NUM:
      rank_image_func = RankNextImageMaxVisiblePointsNum;
      break;
    case Options::ImageSelectionMethod::MAX_VISIBLE_POINTS_RATIO:
      rank_image_func = RankNextImageMaxVisiblePointsRatio;
      break;
    case Options::ImageSelectionMethod::MIN_UNCERTAINTY:
      rank_image_func = RankNextImageMinUncertainty;
      break;
  }

  std::vector<std::pair<image_t, float>> image_ranks;
  std::vector<std::pair<image_t, float>> other_image_ranks;

  // Append images that have not failed to register before.
  for (const auto& image : reconstruction_->Images()) {
    // Skip images that are already registered.
    if (image.second.IsRegistered()) {
      continue;
    }

    // Only consider images with a sufficient number of visible points.
    if (image.second.NumVisiblePoints3D() <
        static_cast<size_t>(options.abs_pose_min_num_inliers)) {
      continue;
    }

    // Only try registration for a certain maximum number of times.
    const size_t num_reg_trials = num_reg_trials_[image.first];
    if (num_reg_trials >= static_cast<size_t>(options.max_reg_trials)) {
      continue;
    }

    // If image has been filtered or failed to register, place it in the
    // second bucket and prefer images that have not been tried before.
    const float rank = rank_image_func(image.second);
    if (filtered_images_.count(image.first) == 0 && num_reg_trials == 0) {
      image_ranks.emplace_back(image.first, rank);
    } else {
      other_image_ranks.emplace_back(image.first, rank);
    }
  }

  std::vector<image_t> ranked_images_ids;
  SortAndAppendNextImages(image_ranks, &ranked_images_ids);
  SortAndAppendNextImages(other_image_ranks, &ranked_images_ids);

  return ranked_images_ids;
}

bool HybridMapper::RegisterInitialImagePair(const Options& options,
                                            const image_t image_id1,
                                            const image_t image_id2) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_EQ(reconstruction_->NumRegImages(), 0);

  CHECK(options.Check());

  init_num_reg_trials_[image_id1] += 1;
  init_num_reg_trials_[image_id2] += 1;
  num_reg_trials_[image_id1] += 1;
  num_reg_trials_[image_id2] += 1;

  const image_pair_t pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);
  init_image_pairs_.insert(pair_id);

  Image& image1 = reconstruction_->Image(image_id1);
  const Camera& camera1 = reconstruction_->Camera(image1.CameraId());

  Image& image2 = reconstruction_->Image(image_id2);
  const Camera& camera2 = reconstruction_->Camera(image2.CameraId());

  //////////////////////////////////////////////////////////////////////////////
  // Estimate two-view geometry
  //////////////////////////////////////////////////////////////////////////////

  if (!EstimateInitialTwoViewGeometry(options, image_id1, image_id2)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Setup inital image pose
  //////////////////////////////////////////////////////////////////////////////

  image1.Qvec() = ComposeIdentityQuaternion();
  image1.Tvec() = Eigen::Vector3d(0, 0, 0);
  if (options.init_with_arkit_pose) {
    const auto& image_1_arkit = reconstruction_->ArkitPose(image_id1).Inverse();
    const auto& image_2_arkit = reconstruction_->ArkitPose(image_id2).Inverse();
    image1.Qvec() = image_1_arkit.qvec;
    image1.Tvec() = image_1_arkit.tvec;
    image2.Qvec() = image_2_arkit.qvec;
    image2.Tvec() = image_2_arkit.tvec;
  } else {
    image2.Qvec() = prev_init_two_view_geometry_.qvec;
    image2.Tvec() = prev_init_two_view_geometry_.tvec;
  }

  const Eigen::Matrix3x4d proj_matrix1 = image1.ProjectionMatrix();
  const Eigen::Matrix3x4d proj_matrix2 = image2.ProjectionMatrix();
  const Eigen::Vector3d proj_center1 = image1.ProjectionCenter();
  const Eigen::Vector3d proj_center2 = image2.ProjectionCenter();

  //////////////////////////////////////////////////////////////////////////////
  // Update Reconstruction
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id1);
  reconstruction_->RegisterImage(image_id2);
  RegisterImageEvent(image_id1);
  RegisterImageEvent(image_id2);

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches& corrs =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  const double min_tri_angle_rad = DegToRad(options.init_min_tri_angle);

  // Add 3D point tracks.
  Track track;
  track.Reserve(2);
  track.AddElement(TrackElement());
  track.AddElement(TrackElement());
  track.Element(0).image_id = image_id1;
  track.Element(1).image_id = image_id2;
  for (const auto& corr : corrs) {
    const Eigen::Vector2d point1_N =
        camera1.ImageToWorld(image1.Point2D(corr.point2D_idx1).XY());
    const Eigen::Vector2d point2_N =
        camera2.ImageToWorld(image2.Point2D(corr.point2D_idx2).XY());
    const Eigen::Vector3d& xyz =
        TriangulatePoint(proj_matrix1, proj_matrix2, point1_N, point2_N);
    const double tri_angle =
        CalculateTriangulationAngle(proj_center1, proj_center2, xyz);
    if (tri_angle >= min_tri_angle_rad &&
        HasPointPositiveDepth(proj_matrix1, xyz) &&
        HasPointPositiveDepth(proj_matrix2, xyz)) {
      track.Element(0).point2D_idx = corr.point2D_idx1;
      track.Element(1).point2D_idx = corr.point2D_idx2;
      reconstruction_->AddPoint3D(xyz, track);
    }
  }

  return true;
}

bool HybridMapper::RegisterNextImage(const Options& options,
                                     const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_GE(reconstruction_->NumRegImages(), 2);

  CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);
  Camera& camera = reconstruction_->Camera(image.CameraId());

  CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

  num_reg_trials_[image_id] += 1;

  // Check if enough 2D-3D correspondences.
  if (image.NumVisiblePoints3D() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Search for 2D-3D correspondences
  //////////////////////////////////////////////////////////////////////////////

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();

  std::vector<std::pair<point2D_t, point3D_t>> tri_corrs;
  std::vector<Eigen::Vector2d> tri_points2D;
  std::vector<Eigen::Vector3d> tri_points3D;

  std::unordered_set<point3D_t> corr_point3D_ids;
  for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
       ++point2D_idx) {
    const Point2D& point2D = image.Point2D(point2D_idx);

    corr_point3D_ids.clear();
    for (const auto& corr :
         correspondence_graph.FindCorrespondences(image_id, point2D_idx)) {
      const Image& corr_image = reconstruction_->Image(corr.image_id);
      if (!corr_image.IsRegistered()) {
        continue;
      }

      const Point2D& corr_point2D = corr_image.Point2D(corr.point2D_idx);
      if (!corr_point2D.HasPoint3D()) {
        continue;
      }

      // Avoid duplicate correspondences.
      if (corr_point3D_ids.count(corr_point2D.Point3DId()) > 0) {
        continue;
      }

      const Camera& corr_camera =
          reconstruction_->Camera(corr_image.CameraId());

      // Avoid correspondences to images with bogus camera parameters.
      if (corr_camera.HasBogusParams(options.min_focal_length_ratio,
                                     options.max_focal_length_ratio,
                                     options.max_extra_param)) {
        continue;
      }

      const Point3D& point3D =
          reconstruction_->Point3D(corr_point2D.Point3DId());

      tri_corrs.emplace_back(point2D_idx, corr_point2D.Point3DId());
      corr_point3D_ids.insert(corr_point2D.Point3DId());
      tri_points2D.push_back(point2D.XY());
      tri_points3D.push_back(point3D.XYZ());
    }
  }

  // The size of `next_image.num_tri_obs` and `tri_corrs_point2D_idxs.size()`
  // can only differ, when there are images with bogus camera parameters, and
  // hence we skip some of the 2D-3D correspondences.
  if (tri_points2D.size() <
      static_cast<size_t>(options.abs_pose_min_num_inliers)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // 2D-3D estimation
  //////////////////////////////////////////////////////////////////////////////

  // Only refine / estimate focal length, if no focal length was specified
  // (manually or through EXIF) and if it was not already estimated previously
  // from another image (when multiple images share the same camera
  // parameters)

  AbsolutePoseEstimationOptions abs_pose_options;
  RANSACOptions abs_pos_ransac_options;
  abs_pose_options.num_threads = options.num_threads;
  abs_pose_options.num_focal_length_samples = 30;
  abs_pose_options.min_focal_length_ratio = options.min_focal_length_ratio;
  abs_pose_options.max_focal_length_ratio = options.max_focal_length_ratio;
  abs_pose_options.ransac_options.max_error = options.abs_pose_max_error;
  abs_pose_options.ransac_options.min_inlier_ratio =
      options.abs_pose_min_inlier_ratio;
  abs_pos_ransac_options.max_error = 4.0;
  abs_pos_ransac_options.min_inlier_ratio = options.abs_pose_min_inlier_ratio;

  // Use high confidence to avoid preemptive termination of P3P RANSAC
  // - too early termination may lead to bad registration.
  abs_pose_options.ransac_options.min_num_trials = 100;
  abs_pose_options.ransac_options.max_num_trials = 10000;
  abs_pose_options.ransac_options.confidence = 0.99999;
  abs_pos_ransac_options.min_num_trials = 100;
  abs_pos_ransac_options.max_num_trials = 1000;
  abs_pos_ransac_options.confidence = 0.99999;

  AbsolutePoseRefinementOptions abs_pose_refinement_options;
  if (num_reg_images_per_camera_[image.CameraId()] > 0) {
    // Camera already refined from another image with the same camera.
    if (camera.HasBogusParams(options.min_focal_length_ratio,
                              options.max_focal_length_ratio,
                              options.max_extra_param)) {
      // Previously refined camera has bogus parameters,
      // so reset parameters and try to re-estimage.
      camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
      abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
      abs_pose_refinement_options.refine_focal_length = true;
      abs_pose_refinement_options.refine_extra_params = true;
    } else {
      abs_pose_options.estimate_focal_length = false;
      abs_pose_refinement_options.refine_focal_length = false;
      abs_pose_refinement_options.refine_extra_params = false;
    }
  } else {
    // Camera not refined before. Note that the camera parameters might have
    // been changed before but the image was filtered, so we explicitly reset
    // the camera parameters and try to re-estimate them.
    camera.SetParams(database_cache_->Camera(image.CameraId()).Params());
    abs_pose_options.estimate_focal_length = !camera.HasPriorFocalLength();
    abs_pose_refinement_options.refine_focal_length = true;
    abs_pose_refinement_options.refine_extra_params = true;
  }

  if (!options.abs_pose_refine_focal_length) {
    abs_pose_options.estimate_focal_length = false;
    abs_pose_refinement_options.refine_focal_length = false;
  }

  if (!options.abs_pose_refine_extra_params) {
    abs_pose_refinement_options.refine_extra_params = false;
  }

  size_t num_inliers;
  std::vector<char> inlier_mask;
  if (!EstimateAbsolutePose(abs_pose_options, tri_points2D, tri_points3D,
                            &image.Qvec(), &image.Tvec(), &camera, &num_inliers,
                            &inlier_mask)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Pose refinement
  //////////////////////////////////////////////////////////////////////////////

  const auto& arkit_poses = reconstruction_->ArkitPoses();
  if (arkit_poses.count(image_id) > 0) {
    const auto& prior_pose = arkit_poses.at(image_id);
    if (!RefineAbsolutePoseWithPrior(abs_pose_refinement_options, inlier_mask,
                                     tri_points2D, tri_points3D,
                                     prior_pose.qvec, prior_pose.tvec,
                                     &image.Qvec(), &image.Tvec(), &camera)) {
      return false;
    }
  } else if (!RefineAbsolutePose(abs_pose_refinement_options, inlier_mask,
                                 tri_points2D, tri_points3D, &image.Qvec(),
                                 &image.Tvec(), &camera)) {
    return false;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Continue tracks
  //////////////////////////////////////////////////////////////////////////////

  reconstruction_->RegisterImage(image_id);
  RegisterImageEvent(image_id);

  for (size_t i = 0; i < inlier_mask.size(); ++i) {
    if (inlier_mask[i]) {
      const point2D_t point2D_idx = tri_corrs[i].first;
      const Point2D& point2D = image.Point2D(point2D_idx);
      if (!point2D.HasPoint3D()) {
        const point3D_t point3D_id = tri_corrs[i].second;
        const TrackElement track_el(image_id, point2D_idx);
        reconstruction_->AddObservation(point3D_id, track_el);
        triangulator_->AddModifiedPoint3D(point3D_id);
      }
    }
  }

  return true;
}

bool HybridMapper::RegisterNextImageWithArkit(const Options& options,
                                              const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_GE(reconstruction_->NumRegImages(), 2);

  CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);

  CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

  num_reg_trials_[image_id] += 1;

  const auto& arkit_pose = reconstruction_->ArkitPose(image_id).Inverse();

  image.Qvec() = arkit_pose.qvec;
  image.Tvec() = arkit_pose.tvec;

  reconstruction_->RegisterImage(image_id);
  RegisterImageEvent(image_id);

  return true;
}

bool HybridMapper::RegisterNextImageWithRGBD(const Options& options,
                                             const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  CHECK_GE(reconstruction_->NumRegImages(), 2);

  CHECK(options.Check());

  Image& image = reconstruction_->Image(image_id);

  CHECK(!image.IsRegistered()) << "Image cannot be registered multiple times";

  num_reg_trials_[image_id] += 1;

  const auto* correspondence_graph = reconstruction_->GetCorrespondenceGraph();

  image_t max_matched_image_id = kInvalidImageId;
  image_t min_knn_image_id = kInvalidImageId;
  int knn_dis = std::numeric_limits<int>::max();
  int max_num_feature_matches = -1;
  for (const image_t& reg_image_id : reconstruction_->RegImageIds()) {
    const auto& feature_matches =
        correspondence_graph->FindCorrespondencesBetweenImages(image_id,
                                                               reg_image_id);
    const int num_feature_matches = static_cast<int>(feature_matches.size());

    if (num_feature_matches > max_num_feature_matches) {
      max_matched_image_id = reg_image_id;
      max_num_feature_matches = num_feature_matches;
    }

    if (std::abs((int)reg_image_id - (int)image_id) < knn_dis) {
      min_knn_image_id = reg_image_id;
    }
  }

  if ((max_matched_image_id == kInvalidImageId ||
       max_num_feature_matches < 10) &&
      min_knn_image_id == kInvalidImageId) {
    return false;
  }

  size_t suit_image_id =
      max_num_feature_matches > 100 ? max_matched_image_id : min_knn_image_id;
  std::cout << "register image: " << image_id
            << ", knn_image: " << min_knn_image_id
            << ", match image: " << max_matched_image_id
            << ", suit image id: " << suit_image_id
            << ", max_matches: " << max_num_feature_matches << std::endl;

  Image& ref_image = reconstruction_->Image(suit_image_id);
  Camera& camera = reconstruction_->Camera(image.CameraId());

  Eigen::Matrix3d Rij;
  Eigen::Vector3d tij;

  if (!EstimateRelativePoseFromRGBD(
          image.RGBPath(), image.DepthPath(), ref_image.RGBPath(),
          ref_image.DepthPath(), camera.FocalLengthX(), camera.FocalLengthY(),
          camera.PrincipalPointX(), camera.PrincipalPointY(), &Rij, &tij)) {
    return false;
  }

  Eigen::Matrix3d Rj = QuaternionToRotationMatrix(ref_image.Qvec());
  Eigen::Vector3d tj = ref_image.Tvec();

  Eigen::Matrix3d Ri = Rij.transpose() * Rj;
  Eigen::Vector3d ti = Rij.transpose() * (tj - tij);

  image.Qvec() = RotationMatrixToQuaternion(Ri);
  image.Tvec() = ti;

  reconstruction_->RegisterImage(image_id);
  RegisterImageEvent(image_id);

  return true;
}

size_t HybridMapper::TriangulateImage(
    const IncrementalTriangulator::Options& tri_options,
    const image_t image_id) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->TriangulateImage(tri_options, image_id);
}

size_t HybridMapper::Retriangulate(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->Retriangulate(tri_options);
}

size_t HybridMapper::CompleteTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->CompleteAllTracks(tri_options);
}

size_t HybridMapper::MergeTracks(
    const IncrementalTriangulator::Options& tri_options) {
  CHECK_NOTNULL(reconstruction_);
  return triangulator_->MergeAllTracks(tri_options);
}

HybridMapper::LocalBundleAdjustmentReport HybridMapper::AdjustLocalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options,
    const IncrementalTriangulator::Options& tri_options, const image_t image_id,
    const std::unordered_set<point3D_t>& point3D_ids) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  LocalBundleAdjustmentReport report;

  // Find images that have most 3D points with given image in common.
  const auto origin_image_id = reconstruction_->RegImageIds()[0];
  const std::vector<image_t> local_bundle = FindLocalBundle(options, image_id);

  // Do the bundle adjustment only if there is any connected images.
  if (local_bundle.size() > 0) {
    BundleAdjustmentConfig ba_config;
    ba_config.AddImage(image_id);
    for (const image_t local_image_id : local_bundle) {
      ba_config.AddImage(local_image_id);
    }

    // Fix the existing images, if option specified.
    if (options.fix_existing_images) {
      for (const image_t local_image_id : local_bundle) {
        if (existing_image_ids_.count(local_image_id)) {
          ba_config.SetConstantPose(local_image_id);
        }
      }
    }

    // Determine which cameras to fix, when not all the registered images
    // are within the current local bundle.
    std::unordered_map<camera_t, size_t> num_images_per_camera;
    for (const image_t image_id : ba_config.Images()) {
      const Image& image = reconstruction_->Image(image_id);
      num_images_per_camera[image.CameraId()] += 1;
    }

    for (const auto& camera_id_and_num_images_pair : num_images_per_camera) {
      const size_t num_reg_images_for_camera =
          num_reg_images_per_camera_.at(camera_id_and_num_images_pair.first);
      if (camera_id_and_num_images_pair.second < num_reg_images_for_camera) {
        ba_config.SetConstantCamera(camera_id_and_num_images_pair.first);
      }
    }

    // Fix 7 DOF to avoid scale/rotation/translation drift in bundle adjustment.
    if (local_bundle.size() == 1) {
      ba_config.SetConstantPose(local_bundle[0]);
      ba_config.SetConstantTvec(image_id, {0});
    } else if (local_bundle.size() > 1) {
      if (ba_config.HasImage(origin_image_id)) {
        ba_config.SetConstantPose(origin_image_id);
        for (const auto& local_image_id : local_bundle) {
          if (local_image_id != origin_image_id &&
              (!options.fix_existing_images ||
               !existing_image_ids_.count(local_image_id))) {
            ba_config.SetConstantTvec(local_image_id, {0});
            break;
          }
        }
      } else {
        const image_t image_id1 = local_bundle[local_bundle.size() - 1];
        const image_t image_id2 = local_bundle[local_bundle.size() - 2];
        ba_config.SetConstantPose(image_id1);
        if (!options.fix_existing_images ||
            !existing_image_ids_.count(image_id2)) {
          ba_config.SetConstantTvec(image_id2, {0});
        }
      }
    }

    // Make sure, we refine all new and short-track 3D points, no matter if
    // they are fully contained in the local image set or not. Do not include
    // long track 3D points as they are usually already very stable and adding
    // to them to bundle adjustment and track merging/completion would slow
    // down the local bundle adjustment significantly.
    std::unordered_set<point3D_t> variable_point3D_ids;
    for (const point3D_t point3D_id : point3D_ids) {
      const Point3D& point3D = reconstruction_->Point3D(point3D_id);
      const size_t kMaxTrackLength = 15;
      if (!point3D.HasError() || point3D.Track().Length() <= kMaxTrackLength) {
        ba_config.AddVariablePoint(point3D_id);
        variable_point3D_ids.insert(point3D_id);
      }
    }

    // Adjust the local bundle.
    BundleAdjuster bundle_adjuster(ba_options, ba_config);
    bundle_adjuster.Solve(reconstruction_);

    report.num_adjusted_observations =
        bundle_adjuster.Summary().num_residuals / 2;

    // Merge refined tracks with other existing points.
    report.num_merged_observations =
        triangulator_->MergeTracks(tri_options, variable_point3D_ids);
    // Complete tracks that may have failed to triangulate before refinement
    // of camera pose and calibration in bundle-adjustment. This may avoid
    // that some points are filtered and it helps for subsequent image
    // registrations.
    report.num_completed_observations =
        triangulator_->CompleteTracks(tri_options, variable_point3D_ids);
    report.num_completed_observations +=
        triangulator_->CompleteImage(tri_options, image_id);
  }

  // Filter both the modified images and all changed 3D points to make sure
  // there are no outlier points in the model. This results in duplicate work as
  // many of the provided 3D points may also be contained in the adjusted
  // images, but the filtering is not a bottleneck at this point.
  std::unordered_set<image_t> filter_image_ids;
  filter_image_ids.insert(image_id);
  filter_image_ids.insert(local_bundle.begin(), local_bundle.end());
  report.num_filtered_observations = reconstruction_->FilterPoints3DInImages(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      filter_image_ids);
  report.num_filtered_observations += reconstruction_->FilterPoints3D(
      options.filter_max_reproj_error, options.filter_min_tri_angle,
      point3D_ids);

  return report;
}

bool HybridMapper::AdjustGlobalBundle(
    const Options& options, const BundleAdjustmentOptions& ba_options) {
  CHECK_NOTNULL(reconstruction_);
  const auto& reg_image_ids = reconstruction_->RegImageIds();

  CHECK_GE(reg_image_ids.size(), 2) << "At least two images must be "
                                       "registered for global "
                                       "bundle-adjustment";

  // Avoid degeneracies in bundle adjustment.
  reconstruction_->FilterObservationsWithNegativeDepth();

  // Configure bundle adjustment.
  BundleAdjustmentConfig ba_config;
  for (const image_t image_id : reg_image_ids) {
    ba_config.AddImage(image_id);
  }

  // Fix the existing images, if option specified.
  if (options.fix_existing_images) {
    for (const image_t image_id : reg_image_ids) {
      if (existing_image_ids_.count(image_id)) {
        ba_config.SetConstantPose(image_id);
      }
    }
  }

  // Fix 7-DOFs of the bundle adjustment problem.
  ba_config.SetConstantPose(reg_image_ids[0]);
  if (!options.depth_available &&
      (!options.fix_existing_images ||
       !existing_image_ids_.count(reg_image_ids[1]))) {
    ba_config.SetConstantTvec(reg_image_ids[1], {0});
  }

  // If no register init pose with arkit, perform sim3.
  if (!options.init_with_arkit_pose) {
  }

  // Run bundle adjustment.
  BundleAdjuster bundle_adjuster(ba_options, ba_config);
  for (const image_t image_id : reg_image_ids) {
    const auto& arkit_pose = reconstruction_->ArkitPose(image_id);
    bundle_adjuster.AddPriorOrientation(image_id, arkit_pose.qvec);
    bundle_adjuster.AddPriorPosition(image_id, arkit_pose.tvec);
  }

  if (!bundle_adjuster.Solve(reconstruction_)) {
    return false;
  }

  return true;
}

size_t HybridMapper::FilterImages(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());

  // Do not filter images in the early stage of the reconstruction, since the
  // calibration is often still refining a lot. Hence, the camera parameters
  // are not stable in the beginning.
  const size_t kMinNumImages = 20;
  if (reconstruction_->NumRegImages() < kMinNumImages) {
    return {};
  }

  const std::vector<image_t> image_ids = reconstruction_->FilterImages(
      options.min_focal_length_ratio, options.max_focal_length_ratio,
      options.max_extra_param);

  for (const image_t image_id : image_ids) {
    DeRegisterImageEvent(image_id);
    filtered_images_.insert(image_id);
  }

  return image_ids.size();
}

size_t HybridMapper::FilterPoints(const Options& options) {
  CHECK_NOTNULL(reconstruction_);
  CHECK(options.Check());
  return reconstruction_->FilterAllPoints3D(options.filter_max_reproj_error,
                                            options.filter_min_tri_angle);
}

const Reconstruction& HybridMapper::GetReconstruction() const {
  return *reconstruction_;
}

size_t HybridMapper::NumTotalRegImages() const { return num_total_reg_images_; }

size_t HybridMapper::NumSharedRegImages() const {
  return num_shared_reg_images_;
}

const std::unordered_set<point3D_t>& HybridMapper::GetModifiedPoints3D() {
  return triangulator_->GetModifiedPoints3D();
}

void HybridMapper::ClearModifiedPoints3D() {
  triangulator_->ClearModifiedPoints3D();
}

std::vector<image_t> HybridMapper::FindFirstInitialImage(
    const Options& options) const {
  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    image_t num_correspondences;
  };

  const size_t init_max_reg_trials =
      static_cast<size_t>(options.init_max_reg_trials);

  // Collect information of all not yet registered images with
  // correspondences.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto& image : reconstruction_->Images()) {
    // Only images with correspondences can be registered.
    if (image.second.NumCorrespondences() == 0) {
      continue;
    }

    // Only use images for initialization a maximum number of times.
    if (init_num_reg_trials_.count(image.first) &&
        init_num_reg_trials_.at(image.first) >= init_max_reg_trials) {
      continue;
    }

    // Only use images for initialization that are not registered in any
    // of the other reconstructions.
    if (num_registrations_.count(image.first) > 0 &&
        num_registrations_.at(image.first) > 0) {
      continue;
    }

    const class Camera& camera =
        reconstruction_->Camera(image.second.CameraId());
    ImageInfo image_info;
    image_info.image_id = image.first;
    image_info.prior_focal_length = camera.HasPriorFocalLength();
    image_info.num_correspondences = image.second.NumCorrespondences();
    image_infos.push_back(image_info);
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> HybridMapper::FindSecondInitialImage(
    const Options& options, const image_t image_id1) const {
  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();

  // Collect images that are connected to the first seed image and have
  // not been registered before in other reconstructions.
  const class Image& image1 = reconstruction_->Image(image_id1);
  std::unordered_map<image_t, point2D_t> num_correspondences;
  for (point2D_t point2D_idx = 0; point2D_idx < image1.NumPoints2D();
       ++point2D_idx) {
    for (const auto& corr :
         correspondence_graph.FindCorrespondences(image_id1, point2D_idx)) {
      if (num_registrations_.count(corr.image_id) == 0 ||
          num_registrations_.at(corr.image_id) == 0) {
        num_correspondences[corr.image_id] += 1;
      }
    }
  }

  // Struct to hold meta-data for ranking images.
  struct ImageInfo {
    image_t image_id;
    bool prior_focal_length;
    point2D_t num_correspondences;
  };

  const size_t init_min_num_inliers =
      static_cast<size_t>(options.init_min_num_inliers);

  // Compose image information in a compact form for sorting.
  std::vector<ImageInfo> image_infos;
  image_infos.reserve(reconstruction_->NumImages());
  for (const auto elem : num_correspondences) {
    if (elem.second >= init_min_num_inliers) {
      const class Image& image = reconstruction_->Image(elem.first);
      const class Camera& camera = reconstruction_->Camera(image.CameraId());
      ImageInfo image_info;
      image_info.image_id = elem.first;
      image_info.prior_focal_length = camera.HasPriorFocalLength();
      image_info.num_correspondences = elem.second;
      image_infos.push_back(image_info);
    }
  }

  // Sort images such that images with a prior focal length and more
  // correspondences are preferred, i.e. they appear in the front of the list.
  std::sort(
      image_infos.begin(), image_infos.end(),
      [](const ImageInfo& image_info1, const ImageInfo& image_info2) {
        if (image_info1.prior_focal_length && !image_info2.prior_focal_length) {
          return true;
        } else if (!image_info1.prior_focal_length &&
                   image_info2.prior_focal_length) {
          return false;
        } else {
          return image_info1.num_correspondences >
                 image_info2.num_correspondences;
        }
      });

  // Extract image identifiers in sorted order.
  std::vector<image_t> image_ids;
  image_ids.reserve(image_infos.size());
  for (const ImageInfo& image_info : image_infos) {
    image_ids.push_back(image_info.image_id);
  }

  return image_ids;
}

std::vector<image_t> HybridMapper::FindLocalBundle(
    const Options& options, const image_t image_id) const {
  CHECK(options.Check());

  const Image& image = reconstruction_->Image(image_id);
  CHECK(image.IsRegistered());

  // Extract all images that have at least one 3D point with the query image
  // in common, and simultaneously count the number of common 3D points.

  std::unordered_map<image_t, size_t> shared_observations;

  std::unordered_set<point3D_t> point3D_ids;
  point3D_ids.reserve(image.NumPoints3D());

  for (const Point2D& point2D : image.Points2D()) {
    if (point2D.HasPoint3D()) {
      point3D_ids.insert(point2D.Point3DId());
      const Point3D& point3D = reconstruction_->Point3D(point2D.Point3DId());
      for (const TrackElement& track_el : point3D.Track().Elements()) {
        if (track_el.image_id != image_id) {
          shared_observations[track_el.image_id] += 1;
        }
      }
    }
  }

  // Sort overlapping images according to number of shared observations.

  std::vector<std::pair<image_t, size_t>> overlapping_images(
      shared_observations.begin(), shared_observations.end());
  std::sort(overlapping_images.begin(), overlapping_images.end(),
            [](const std::pair<image_t, size_t>& image1,
               const std::pair<image_t, size_t>& image2) {
              return image1.second > image2.second;
            });

  // The local bundle is composed of the given image and its most connected
  // neighbor images, hence the subtraction of 1.

  const size_t num_images =
      static_cast<size_t>(options.local_ba_num_images - 1);
  const size_t num_eff_images = std::min(num_images, overlapping_images.size());

  // Extract most connected images and ensure sufficient triangulation angle.

  std::vector<image_t> local_bundle_image_ids;
  local_bundle_image_ids.reserve(num_eff_images);

  // If the number of overlapping images equals the number of desired images in
  // the local bundle, then simply copy over the image identifiers.
  if (overlapping_images.size() == num_eff_images) {
    for (const auto& overlapping_image : overlapping_images) {
      local_bundle_image_ids.push_back(overlapping_image.first);
    }
    return local_bundle_image_ids;
  }

  // In the following iteration, we start with the most overlapping images and
  // check whether it has sufficient triangulation angle. If none of the
  // overlapping images has sufficient triangulation angle, we relax the
  // triangulation angle threshold and start from the most overlapping image
  // again. In the end, if we still haven't found enough images, we simply use
  // the most overlapping images.

  const double min_tri_angle_rad = DegToRad(options.local_ba_min_tri_angle);

  // The selection thresholds (minimum triangulation angle, minimum number of
  // shared observations), which are successively relaxed.
  const std::array<std::pair<double, double>, 8> selection_thresholds = {{
      std::make_pair(min_tri_angle_rad / 1.0, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 1.5, 0.6 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.0, 0.5 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 2.5, 0.4 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 3.0, 0.3 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 4.0, 0.2 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 5.0, 0.1 * image.NumPoints3D()),
      std::make_pair(min_tri_angle_rad / 6.0, 0.1 * image.NumPoints3D()),
  }};

  const Eigen::Vector3d proj_center = image.ProjectionCenter();
  std::vector<Eigen::Vector3d> shared_points3D;
  shared_points3D.reserve(image.NumPoints3D());
  std::vector<double> tri_angles(overlapping_images.size(), -1.0);
  std::vector<char> used_overlapping_images(overlapping_images.size(), false);

  for (const auto& selection_threshold : selection_thresholds) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Check if the image has sufficient overlap. Since the images are ordered
      // based on the overlap, we can just skip the remaining ones.
      if (overlapping_images[overlapping_image_idx].second <
          selection_threshold.second) {
        break;
      }

      // Check if the image is already in the local bundle.
      if (used_overlapping_images[overlapping_image_idx]) {
        continue;
      }

      const auto& overlapping_image = reconstruction_->Image(
          overlapping_images[overlapping_image_idx].first);
      const Eigen::Vector3d overlapping_proj_center =
          overlapping_image.ProjectionCenter();

      // In the first iteration, compute the triangulation angle. In later
      // iterations, reuse the previously computed value.
      double& tri_angle = tri_angles[overlapping_image_idx];
      if (tri_angle < 0.0) {
        // Collect the commonly observed 3D points.
        shared_points3D.clear();
        for (const Point2D& point2D : image.Points2D()) {
          if (point2D.HasPoint3D() && point3D_ids.count(point2D.Point3DId())) {
            shared_points3D.push_back(
                reconstruction_->Point3D(point2D.Point3DId()).XYZ());
          }
        }

        // Calculate the triangulation angle at a certain percentile.
        const double kTriangulationAnglePercentile = 75;
        tri_angle = Percentile(
            CalculateTriangulationAngles(proj_center, overlapping_proj_center,
                                         shared_points3D),
            kTriangulationAnglePercentile);
      }

      // Check that the image has sufficient triangulation angle.
      if (tri_angle >= selection_threshold.first) {
        local_bundle_image_ids.push_back(overlapping_image.ImageId());
        used_overlapping_images[overlapping_image_idx] = true;
        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }

    // Check if we already collected enough images.
    if (local_bundle_image_ids.size() >= num_eff_images) {
      break;
    }
  }

  // In case there are not enough images with sufficient triangulation angle,
  // simply fill up the rest with the most overlapping images.

  if (local_bundle_image_ids.size() < num_eff_images) {
    for (size_t overlapping_image_idx = 0;
         overlapping_image_idx < overlapping_images.size();
         ++overlapping_image_idx) {
      // Collect image if it is not yet in the local bundle.
      if (!used_overlapping_images[overlapping_image_idx]) {
        local_bundle_image_ids.push_back(
            overlapping_images[overlapping_image_idx].first);
        used_overlapping_images[overlapping_image_idx] = true;

        // Check if we already collected enough images.
        if (local_bundle_image_ids.size() >= num_eff_images) {
          break;
        }
      }
    }
  }

  return local_bundle_image_ids;
}

void HybridMapper::RegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_[image.CameraId()];
  num_reg_images_for_camera += 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image += 1;
  if (num_regs_for_image == 1) {
    num_total_reg_images_ += 1;
  } else if (num_regs_for_image > 1) {
    num_shared_reg_images_ += 1;
  }
}

void HybridMapper::DeRegisterImageEvent(const image_t image_id) {
  const Image& image = reconstruction_->Image(image_id);
  size_t& num_reg_images_for_camera =
      num_reg_images_per_camera_.at(image.CameraId());
  CHECK_GT(num_reg_images_for_camera, 0);
  num_reg_images_for_camera -= 1;

  size_t& num_regs_for_image = num_registrations_[image_id];
  num_regs_for_image -= 1;
  if (num_regs_for_image == 0) {
    num_total_reg_images_ -= 1;
  } else if (num_regs_for_image > 0) {
    num_shared_reg_images_ -= 1;
  }
}

bool HybridMapper::EstimateInitialTwoViewGeometry(const Options& options,
                                                  const image_t image_id1,
                                                  const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);

  if (prev_init_image_pair_id_ == image_pair_id) {
    return true;
  }

  const Image& image1 = database_cache_->Image(image_id1);
  const Camera& camera1 = database_cache_->Camera(image1.CameraId());

  const Image& image2 = database_cache_->Image(image_id2);
  const Camera& camera2 = database_cache_->Camera(image2.CameraId());

  const CorrespondenceGraph& correspondence_graph =
      database_cache_->CorrespondenceGraph();
  const FeatureMatches matches =
      correspondence_graph.FindCorrespondencesBetweenImages(image_id1,
                                                            image_id2);

  std::vector<Eigen::Vector2d> points1;
  points1.reserve(image1.NumPoints2D());
  for (const auto& point : image1.Points2D()) {
    points1.push_back(point.XY());
  }

  std::vector<Eigen::Vector2d> points2;
  points2.reserve(image2.NumPoints2D());
  for (const auto& point : image2.Points2D()) {
    points2.push_back(point.XY());
  }

  TwoViewGeometry two_view_geometry;
  TwoViewGeometry::Options two_view_geometry_options;
  two_view_geometry_options.ransac_options.min_num_trials = 30;
  two_view_geometry_options.ransac_options.max_error = options.init_max_error;
  two_view_geometry.EstimateCalibrated(camera1, points1, camera2, points2,
                                       matches, two_view_geometry_options);

  if (!two_view_geometry.EstimateRelativePose(camera1, points1, camera2,
                                              points2)) {
    return false;
  }

  if (static_cast<int>(two_view_geometry.inlier_matches.size()) >=
          options.init_min_num_inliers &&
      std::abs(two_view_geometry.tvec.z()) < options.init_max_forward_motion &&
      two_view_geometry.tri_angle > DegToRad(options.init_min_tri_angle)) {
    prev_init_image_pair_id_ = image_pair_id;
    prev_init_two_view_geometry_ = two_view_geometry;
    return true;
  }

  return false;
}

bool HybridMapper::EstimateInitialRelativePose(const Options& options,
                                               const image_t image_id1,
                                               const image_t image_id2) {
  const image_pair_t image_pair_id =
      Database::ImagePairToPairId(image_id1, image_id2);

  if (prev_init_image_pair_id_ == image_pair_id) {
    return true;
  }

  const Image& image1 = database_cache_->Image(image_id1);
  const Camera& camera1 = database_cache_->Camera(image1.CameraId());

  const Image& image2 = database_cache_->Image(image_id2);
  const Camera& camera2 = database_cache_->Camera(image2.CameraId());

  if (camera1.CameraId() != camera2.CameraId()) {
    LOG(WARNING) << "Estimte Reltive Pose from image pair need use the same "
                    "camera model";
    return false;
  }

  Eigen::Matrix3d rotation_1_to_2 = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation_1_to_2 = Eigen::Vector3d::Zero();
  // TODO 增加RGBD图像不确定性的分析
  if (EstimateRelativePoseFromRGBD(
          image1.RGBPath(), image1.DepthPath(), image2.RGBPath(),
          image2.DepthPath(), camera1.FocalLengthX(), camera1.FocalLengthY(),
          camera1.PrincipalPointX(), camera1.PrincipalPointY(),
          &rotation_1_to_2, &translation_1_to_2)) {
    prev_init_image_pair_id_ = image_pair_id;
    prev_init_two_view_geometry_.qvec =
        RotationMatrixToQuaternion(rotation_1_to_2);
    prev_init_two_view_geometry_.tvec = translation_1_to_2;

    return true;
  }

  return false;
}

std::unordered_set<image_t> HybridMapper::RemoveDisconnectedView(
    ViewGraph* view_graph) {
  CHECK_NOTNULL(view_graph);
  std::unordered_set<image_t> removed_views;

  // Extractor all connected components.
  ConnectedComponents<image_t> cc_extractor;
  const auto& view_pairs = view_graph->GetAllEdges();
  for (const auto& view_pair : view_pairs) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(view_pair.first, &image_id1, &image_id2);
    cc_extractor.AddEdge(image_id1, image_id2);
  }
  std::unordered_map<image_t, std::unordered_set<image_t>> connected_components;
  cc_extractor.Extract(&connected_components);

  // Find the largest connecte component
  size_t max_cc_size = 0;
  image_t largest_cc_root_id = kInvalidImageId;
  for (const auto& connected_component : connected_components) {
    if (connected_component.second.size() > max_cc_size) {
      max_cc_size = connected_component.second.size();
      largest_cc_root_id = connected_component.first;
    }
  }

  std::cout << "  => connected components size: " << connected_components.size()
            << std::endl;

  const int num_view_pairs_before_filtering = view_graph->NumEdges();
  for (const auto& connected_component : connected_components) {
    if (connected_component.first == largest_cc_root_id) {
      continue;
    }

    for (const image_t view_id2 : connected_component.second) {
      view_graph->RemoveImage(view_id2);
      removed_views.insert(view_id2);
    }
  }

  const int num_removed_view_pairs =
      num_view_pairs_before_filtering - view_graph->NumEdges();

  LOG_IF(INFO, num_removed_view_pairs > 0)
      << num_removed_view_pairs
      << " view pairs were disconnected from the largest connected component "
         "of the view graph and were removed.";
  return removed_views;
}

bool HybridMapper::OrientationsFromMaximumSpanningTree(
    const ViewGraph& view_graph,
    std::unordered_map<image_t, Eigen::Vector3d>* orientations) {
  CHECK_NOTNULL(orientations);

  std::unordered_set<image_t> largest_cc;
  view_graph.GetLargestConnectedComponentIds(&largest_cc);
  ViewGraph largest_cc_sub_graph;
  view_graph.ExtractSubgraph(largest_cc, &largest_cc_sub_graph);

  // Compute maximum spanning tree
  const auto& all_edges = largest_cc_sub_graph.GetAllEdges();
  MinimumSpanningTree<image_t, int> mst_extractor;
  for (const auto& edges : all_edges) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(edges.first, &image_id1, &image_id2);
    for (const auto& edge : edges.second) {
      if (edge.config == TwoViewGeometry::ARKIT ||
          edge.config == TwoViewGeometry::RGBD) {
        // If the edge is Arkit we Maximum number of features to detect.
        // see feature/sift.h/SiftExtractionOptions
        mst_extractor.AddEdge(image_id1, image_id2,
                              -static_cast<int>(8192 + image_id1 + image_id2));
        continue;
      }

      mst_extractor.AddEdge(image_id1, image_id2,
                            -static_cast<int>(edge.inlier_matches.size()));
    }
  }

  std::unordered_set<std::pair<image_t, image_t>> mst;
  if (!mst_extractor.Extract(&mst)) {
    LOG(WARNING)
        << "Could not extract the maximum spanning tree from the view graph";
    return false;
  }

  // Create an MST view graph
  ViewGraph mst_view_graph;
  for (const auto& view_pair : mst) {
    const auto* edges = view_graph.GetEdge(view_pair.first, view_pair.second);
    for (const auto& edge : *edges) {
      mst_view_graph.AddEdge(view_pair.first, view_pair.second, edge);
    }
  }

  std::vector<HeapElement> heap;

  const image_t root_view_id = mst.begin()->first;

  (*orientations)[root_view_id] = Eigen::Vector3d::Zero();
  AddEdgesToHeap(mst_view_graph, *orientations, root_view_id, &heap);

  while (!heap.empty()) {
    const HeapElement next_edge = heap.front();
    std::pop_heap(heap.begin(), heap.end(), SortHeapElement);
    heap.pop_back();

    if (ContainsKey(*orientations, next_edge.second.second)) {
      continue;
    }

    (*orientations)[next_edge.second.second] = ComputeOrientation(
        FindOrDie(*orientations, next_edge.second.first), next_edge.first,
        next_edge.second.first, next_edge.second.second);

    AddEdgesToHeap(mst_view_graph, *orientations, next_edge.second.second,
                   &heap);
  }

  return true;
}

void HybridMapper::FilterViewPairsFromOrientation(const double max_rot_change) {
  // Precompute the squared threshold in radians.
  const double max_rot_change_rad = DegToRad(max_rot_change);
  const double sq_max_rot_change_rad = max_rot_change_rad * max_rot_change_rad;
  ViewGraph* view_graph = reconstruction_->GetViewGraph();
  std::unordered_set<image_pair_t> view_pairs_to_remove;
  auto& view_pairs = view_graph->GetAllEdges();
  for (auto& view_pair : view_pairs) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(view_pair.first, &image_id1, &image_id2);
    const Eigen::Vector3d* orientation1 =
        FindOrNull(reconstruction_->GlobalOrientations(), image_id1);
    const Eigen::Vector3d* orientation2 =
        FindOrNull(reconstruction_->GlobalOrientations(), image_id2);
    if (orientation1 == nullptr || orientation2 == nullptr) {
      std::cout
          << "WARN View pair (" << image_id1 << ", " << image_id2
          << ") contains a view that does not exist! Removing the view pair.";
      view_pairs_to_remove.insert(view_pair.first);
      continue;
    }

    std::vector<TwoViewGeometry> valid_views;
    for (auto iter = view_pair.second.begin(); iter != view_pair.second.end();
         ++iter) {
      Eigen::Vector3d relative_rotation;
      ceres::QuaternionToAngleAxis(iter->qvec.data(), relative_rotation.data());
      if (AngularDifferenceIsAcceptable(*orientation1, *orientation2,
                                        relative_rotation,
                                        sq_max_rot_change_rad)) {
        valid_views.emplace_back(*iter);
      }
    }

    view_pair.second = valid_views;

    if (view_pair.second.empty()) {
      view_pairs_to_remove.insert(view_pair.first);
    }
  }

  for (const image_pair_t view_id_pair : view_pairs_to_remove) {
    image_t image_id1, image_id2;
    Database::PairIdToImagePair(view_id_pair, &image_id1, &image_id2);
    view_graph->RemoveEdge(image_id1, image_id2);
  }

  VLOG(1) << "Removed " << view_pairs_to_remove.size()
          << " view pairs by rotation filtering.";
}

}  // namespace colmap