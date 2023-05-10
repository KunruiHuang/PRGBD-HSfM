#ifndef COLMAP_HYBRID_MAPPER_H
#define COLMAP_HYBRID_MAPPER_H

#include "base/database.h"
#include "base/database_cache.h"
#include "base/reconstruction.h"
#include "optim/bundle_adjustment.h"
#include "sfm/incremental_triangulator.h"
#include "util/alignment.h"

namespace colmap {

class HybridMapper {
 public:
  struct Options {
    // Method to find and select next best image to register.
    enum class ImageSelectionMethod {
      MAX_VISIBLE_POINTS_NUM,
      MAX_VISIBLE_POINTS_RATIO,
      MIN_UNCERTAINTY,
    };

    // Minimum number of inliers for initial image pair.
    int init_min_num_inliers = 100;

    // Maximum error in pixels for two-view geometry estimation for initial
    // image pair.
    double init_max_error = 4.0;

    // Maximum forward motion for initial image pair.
    double init_max_forward_motion = 0.95;

    // Minimum triangulation angle for initial image pair.
    double init_min_tri_angle = 16.0;

    // Maximum number of trials to use an image for initialization.
    int init_max_reg_trials = 2;

    // Initialize image pair with arkit pose
    bool init_with_arkit_pose = true;

    // Whether to have the depth information in the reconstruction.
    bool depth_available = true;

    // Maximum reprojection error in absolute pose estimation.
    double abs_pose_max_error = 12.0;

    // Minimum number of inliers in absolute pose estimation.
    int abs_pose_min_num_inliers = 30;

    // Minimum inlier ratio in absolute pose estimation.
    double abs_pose_min_inlier_ratio = 0.25;

    // Whether to estimate the focal length in absolute pose estimation.
    bool abs_pose_refine_focal_length = true;

    // Whether to estimate the extra parameters in absolute pose estimation.
    bool abs_pose_refine_extra_params = true;

    // Number of images to optimize in local bundle adjustment.
    int local_ba_num_images = 6;

    // Minimum triangulation for images to be chosen in local bundle adjustment.
    double local_ba_min_tri_angle = 6;

    // Thresholds for bogus camera parameters. Images with bogus camera
    // parameters are filtered and ignored in triangulation.
    double min_focal_length_ratio = 0.1;  // Opening angle of ~130deg
    double max_focal_length_ratio = 10;   // Opening angle of ~5deg
    double max_extra_param = 1;

    // Maximum reprojection error in pixels for observations.
    double filter_max_reproj_error = 4.0;

    // Minimum triangulation angle in degrees for stable 3D points.
    double filter_min_tri_angle = 1.5;

    // Maximum number of trials to register an image.
    int max_reg_trials = 3;

    // The minmum of two view geometry correspondence pair
    int min_num_two_view_inliers = 30;

    // If reconstruction is provided as input, fix the existing image poses.
    bool fix_existing_images = false;

    // check the relative rotation is normal
    double max_relative_rotation_difference = 10.0;

    // Number of threads.
    int num_threads = -1;

    ImageSelectionMethod image_selection_method =
        ImageSelectionMethod::MIN_UNCERTAINTY;

    bool Check() const;
  };

  struct LocalBundleAdjustmentReport {
    size_t num_merged_observations = 0;
    size_t num_completed_observations = 0;
    size_t num_filtered_observations = 0;
    size_t num_adjusted_observations = 0;
  };

  explicit HybridMapper(DatabaseCache* database_cache);

  void BeginReconstruction(Reconstruction* reconstruction);

  void SetUpReconstruction(Reconstruction* reconstruction);

  void EndReconstruction(const bool discard);

  bool FilterInitialViewGraph(const int min_num_two_view_inliers);

  bool EstimateGlobalRotations();

  void FilterRotations(const double max_relative_rotation_difference_degrees);

  bool FindInitialImagePair(const Options& options, image_t* image_id1,
                            image_t* image_id2);

  std::vector<image_t> FindNextImages(const Options& options);

  bool RegisterInitialImagePair(const Options& options, const image_t image_id1,
                                const image_t image_id2);

  bool RegisterNextImage(const Options& options, const image_t image_id);

  bool RegisterNextImageWithArkit(const Options& options,
                                  const image_t image_id);

  bool RegisterNextImageWithRGBD(const Options& options,
                                 const image_t image_id);

  size_t TriangulateImage(const IncrementalTriangulator::Options& tri_options,
                          const image_t image_id);

  size_t Retriangulate(const IncrementalTriangulator::Options& tri_options);

  size_t CompleteTracks(const IncrementalTriangulator::Options& tri_options);

  size_t MergeTracks(const IncrementalTriangulator::Options& tri_options);

  LocalBundleAdjustmentReport AdjustLocalBundle(
      const Options& options, const BundleAdjustmentOptions& ba_options,
      const IncrementalTriangulator::Options& tri_options,
      const image_t image_id, const std::unordered_set<point3D_t>& point3D_ids);

  bool AdjustGlobalBundle(const Options& options,
                          const BundleAdjustmentOptions& ba_options);

  size_t FilterImages(const Options& options);
  size_t FilterPoints(const Options& options);

  const Reconstruction& GetReconstruction() const;

  size_t NumTotalRegImages() const;

  size_t NumSharedRegImages() const;

  const std::unordered_set<point3D_t>& GetModifiedPoints3D();

  void ClearModifiedPoints3D();

 private:
  // Find seed images for incremental reconstruction. Suitable seed images have
  // a large number of correspondences and have camera calibration priors. The
  // returned list is ordered such that most suitable images are in the front.
  std::vector<image_t> FindFirstInitialImage(const Options& options) const;

  // For a given first seed image, find other images that are connected to the
  // first image. Suitable second images have a large number of correspondences
  // to the first image and have camera calibration priors. The returned list is
  // ordered such that most suitable images are in the front.
  std::vector<image_t> FindSecondInitialImage(const Options& options,
                                              const image_t image_id1) const;

  // Find local bundle for given image in the reconstruction. The local bundle
  // is defined as the images that are most connected, i.e. maximum number of
  // shared 3D points, to the given image.
  std::vector<image_t> FindLocalBundle(const Options& options,
                                       const image_t image_id) const;

  // Register / De-register image in current reconstruction and update
  // the number of shared images between all reconstructions.
  void RegisterImageEvent(const image_t image_id);
  void DeRegisterImageEvent(const image_t image_id);

  bool EstimateInitialTwoViewGeometry(const Options& options,
                                      const image_t image_id1,
                                      const image_t image_id2);

  bool EstimateInitialRelativePose(const Options& options,
                                   const image_t image_id1,
                                   const image_t image_id2);

  std::unordered_set<image_t> RemoveDisconnectedView(ViewGraph* view_graph);

  bool OrientationsFromMaximumSpanningTree(
      const ViewGraph& view_graph,
      std::unordered_map<image_t, Eigen::Vector3d>* orientations);

  void FilterViewPairsFromOrientation(const double max_rot_change);

  // Class that holds all necessary data from database in memory.
  DatabaseCache* database_cache_;

  // Class that holds data of the reconstruction.
  Reconstruction* reconstruction_;

  // Class that is responsible for incremental triangulation.
  std::unique_ptr<IncrementalTriangulator> triangulator_;

  // Number of images that are registered in at least on reconstruction.
  size_t num_total_reg_images_;

  // Number of shared images between current reconstruction and all other
  // previous reconstructions.
  size_t num_shared_reg_images_;

  // Estimated two-view geometry of last call to `FindFirstInitialImage`,
  // used as a cache for a subsequent call to `RegisterInitialImagePair`.
  image_pair_t prev_init_image_pair_id_;
  TwoViewGeometry prev_init_two_view_geometry_;

  // Images and image pairs that have been used for initialization. Each image
  // and image pair is only tried once for initialization.
  std::unordered_map<image_t, size_t> init_num_reg_trials_;
  std::unordered_set<image_pair_t> init_image_pairs_;

  // The number of registered images per camera. This information is used
  // to avoid duplicate refinement of camera parameters and degradation of
  // already refined camera parameters in local bundle adjustment when multiple
  // images share intrinsics.
  std::unordered_map<camera_t, size_t> num_reg_images_per_camera_;

  // The number of reconstructions in which images are registered.
  std::unordered_map<image_t, size_t> num_registrations_;

  // Images that have been filtered in current reconstruction.
  std::unordered_set<image_t> filtered_images_;

  // Number of trials to register image in current reconstruction. Used to set
  // an upper bound to the number of trials to register an image.
  std::unordered_map<image_t, size_t> num_reg_trials_;

  // Images that were registered before beginning the reconstruction.
  // This image list will be non-empty, if the reconstruction is continued from
  // an existing reconstruction.
  std::unordered_set<image_t> existing_image_ids_;

  // Save the global estimation result.
  //  std::unordered_map<image_t, Eigen::Vector3d> orientations_;
  //  std::unordered_map<image_t, Eigen::Vector3d> positions_;
};

}  // namespace colmap

#endif  // COLMAP_HYBRID_MAPPER_H
