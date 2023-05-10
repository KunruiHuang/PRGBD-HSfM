#ifndef COLMAP_SRC_CONTROLLERS_HYBRID_MAPPER_H
#define COLMAP_SRC_CONTROLLERS_HYBRID_MAPPER_H

#include "base/reconstruction_manager.h"
#include "base/view_graph.h"
#include "sfm/hybrid_mapper.h"
#include "util/threading.h"

namespace colmap {

struct HybridMapperOptions {
 public:
  // The minimum number of matches for inlier matches to be considered.
  int min_num_matches = 15;

  // Whether to ignore the inlier matches of watermark image pairs.
  bool ignore_watermarks = false;

  // Whether to reconstruct multiple sub-models.
  bool multiple_models = true;

  // The number of sub-models to reconstruct.
  int max_num_models = 50;

  // The maximum number of overlapping images between sub-models. If the
  // current sub-models shares more than this number of images with another
  // model, then the reconstruction is stopped.
  int max_model_overlap = 20;

  // The minimum number of registered images of a sub-model, otherwise the
  // sub-model is discarded.
  int min_model_size = 10;

  // The image identifiers used to initialize the reconstruction. Note that
  // only one or both image identifiers can be specified. In the former case,
  // the second image is automatically determined.
  int init_image_id1 = -1;
  int init_image_id2 = -1;

  // The number of trials to initialize the reconstruction.
  int init_num_trials = 200;

  // Whether to extract colors for reconstructed points.
  bool extract_colors = true;

  // The number of threads to use during reconstruction.
  int num_threads = -1;

  // Thresholds for filtering images with degenerate intrinsics.
  double min_focal_length_ratio = 0.1;
  double max_focal_length_ratio = 10.0;
  double max_extra_param = 1.0;

  // The sqrt information matrix for ba prior 
  double ba_arkit_rotation_weight = 3.0;
  double ba_arkit_position_weight = 3.0; 
  double ba_global_rotation_weight = 0.02;  
  double ba_depth_weight = 0.3;

  // Which intrinsic parameters to optimize during the reconstruction.
  bool ba_refine_focal_length = true;
  bool ba_refine_principal_point = false;
  bool ba_refine_extra_params = true;

  // The minimum number of residuals per bundle adjustment problem to
  // enable multi-threading solving of the problems.
  int ba_min_num_residuals_for_multi_threading = 50000;

  // The number of images to optimize in local bundle adjustment.
  int ba_local_num_images = 6;

  // Ceres solver function tolerance for local bundle adjustment
  double ba_local_function_tolerance = 0.0;

  // The maximum number of local bundle adjustment iterations.
  int ba_local_max_num_iterations = 25;

  // Whether to use PBA in global bundle adjustment.
  bool ba_global_use_pba = false;

  // The GPU index for PBA bundle adjustment.
  int ba_global_pba_gpu_index = -1;

  // The growth rates after which to perform global bundle adjustment.
  double ba_global_images_ratio = 1.1;
  double ba_global_points_ratio = 1.1;
  int ba_global_images_freq = 500;
  int ba_global_points_freq = 250000;

  // Ceres solver function tolerance for global bundle adjustment
  double ba_global_function_tolerance = 0.0;

  // The maximum number of global bundle adjustment iterations.
  int ba_global_max_num_iterations = 50;

  // The thresholds for iterative bundle adjustment refinements.
  int ba_local_max_refinements = 2;
  double ba_local_max_refinement_change = 0.001;
  int ba_global_max_refinements = 5;
  double ba_global_max_refinement_change = 0.0005;

  // Path to a folder with reconstruction snapshots during incremental
  // reconstruction. Snapshots will be saved according to the specified
  // frequency of registered images.
  std::string snapshot_path = "";
  int snapshot_images_freq = 0;

  // Path to image json file with reconstruction with arkit and depth
  std::string image_json;

  // Path to image and depth csv file
  std::string image_depth_csv;
  std::string arkit_pose_csv;

  // Whether to only use Arkit Pose to build view graph
  bool arkit_view_graph = true;
  bool only_use_arkit_pose = false;

  // Whether to try register lower overlapping image with arkit pose
  bool register_image_with_arkit = true;

  // Whether to try register lower overlapping image with rgbd depth.
  bool register_image_with_rgbd = false;

  // Which images to reconstruct. If no images are specified, all images will
  // be reconstructed by default.
  std::unordered_set<std::string> image_names;

  // If reconstruction is provided as input, fix the existing image poses.
  bool fix_existing_images = false;

  HybridMapper::Options Mapper() const;
  IncrementalTriangulator::Options Triangulation() const;
  BundleAdjustmentOptions LocalBundleAdjustment() const;
  BundleAdjustmentOptions GlobalBundleAdjustment() const;

  bool Check() const;

 private:
  friend class OptionManager;
  friend class MapperGeneralOptionsWidget;
  friend class MapperTriangulationOptionsWidget;
  friend class MapperRegistrationOptionsWidget;
  friend class MapperInitializationOptionsWidget;
  friend class MapperBundleAdjustmentOptionsWidget;
  friend class MapperFilteringOptionsWidget;
  friend class ReconstructionOptionsWidget;
  HybridMapper::Options mapper;
  IncrementalTriangulator::Options triangulation;
};

class HybridMapperController : public Thread {
 public:
  enum {
    GLOBAL_ROTATION_CALLBACK,
    GLOBAL_POSITION_CALLBACK,
    INITIAL_IMAGE_PAIR_REG_CALLBACK,
    NEXT_IMAGE_REG_CALLBACK,
    LAST_IMAGE_REG_CALLBACK,
    FINISH_RECONSTRUCTION_CALLBACK
  };

  HybridMapperController(const HybridMapperOptions* options,
                         const std::string& image_path,
                         const std::string& depth_path,
                         const std::string& database_path,
                         ReconstructionManager* reconstruction_manager);

 private:
  void Run() override;
  bool LoadDatabase();
  void Reconstruct(const HybridMapper::Options& init_mapper_options);

  const HybridMapperOptions* options_;
  const std::string image_path_;
  const std::string depth_path_;
  const std::string database_path_;
  ReconstructionManager* reconstruction_manager_;
  DatabaseCache database_cache_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_CONTROLLERS_HYBRID_MAPPER_H
