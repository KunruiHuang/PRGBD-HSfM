// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "base/database_cache.h"

#include <nlohmann/json.hpp>
#include <unordered_set>

#include "base/common.h"
#include "base/pose.h"
#include "feature/utils.h"
#include "rgbd/rgbd.h"
#include "util/csv.h"
#include "util/misc.h"
#include "util/string.h"
#include "util/timer.h"

using json = nlohmann::json;

namespace colmap {

DatabaseCache::DatabaseCache() {}

void DatabaseCache::AddCamera(class Camera camera) {
  const camera_t camera_id = camera.CameraId();
  CHECK(!ExistsCamera(camera_id));
  cameras_.emplace(camera_id, std::move(camera));
}

void DatabaseCache::AddImage(class Image image) {
  const image_t image_id = image.ImageId();
  CHECK(!ExistsImage(image_id));
  correspondence_graph_.AddImage(image_id, image.NumPoints2D());
  images_.emplace(image_id, std::move(image));
}

void DatabaseCache::Load(const Database& database, const size_t min_num_matches,
                         const bool ignore_watermarks,
                         const std::unordered_set<std::string>& image_names) {
  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;

  timer.Start();
  std::cout << "Loading cameras..." << std::flush;

  {
    std::vector<class Camera> cameras = database.ReadAllCameras();
    cameras_.reserve(cameras.size());
    for (auto& camera : cameras) {
      const camera_t camera_id = camera.CameraId();
      cameras_.emplace(camera_id, std::move(camera));
    }
  }

  std::cout << StringPrintf(" %d in %.3fs", cameras_.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading matches..." << std::flush;

  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  std::unordered_map<image_pair_t, int> map_image_pair_ids;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries,
                                 &map_image_pair_ids);

  std::cout << StringPrintf(" %d in %.3fs", image_pair_ids.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  auto UseInlierMatchesCheck = [min_num_matches, ignore_watermarks](
                                   const TwoViewGeometry& two_view_geometry) {
    return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >=
               min_num_matches &&
           (!ignore_watermarks ||
            two_view_geometry.config != TwoViewGeometry::WATERMARK);
  };

  //////////////////////////////////////////////////////////////////////////////
  // Load images
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading images..." << std::flush;

  std::unordered_set<image_t> image_ids;

  {
    std::vector<class Image> images = database.ReadAllImages();
    const size_t num_images = images.size();

    // Determines for which images data should be loaded.
    if (image_names.empty()) {
      for (const auto& image : images) {
        image_ids.insert(image.ImageId());
      }
    } else {
      for (const auto& image : images) {
        if (image_names.count(image.Name()) > 0) {
          image_ids.insert(image.ImageId());
        }
      }
    }

    // Collect all images that are connected in the correspondence graph.
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(image_ids.size());
    for (size_t i = 0; i < image_pair_ids.size(); ++i) {
      if (UseInlierMatchesCheck(two_view_geometries[i])) {
        image_t image_id1;
        image_t image_id2;
        Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
        if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
          connected_image_ids.insert(image_id1);
          connected_image_ids.insert(image_id2);
        }
      }
    }

    // Load images with correspondences and discard images without
    // correspondences, as those images are useless for SfM.
    images_.reserve(connected_image_ids.size());
    for (auto& image : images) {
      const image_t image_id = image.ImageId();
      if (image_ids.count(image_id) > 0 &&
          connected_image_ids.count(image_id) > 0) {
        images_.emplace(image_id, std::move(image));
        const auto keypoints_and_depths =
            database.ReadKeypointsAndDepths(image_id);
        // 设定当前图像的特征点坐标以及depth
        const auto point2ds = FeatureKeypointsAndDepthsToPoint2Ds(
            keypoints_and_depths.first, keypoints_and_depths.second);
        images_[image_id].SetPoints2D(point2ds);
      }
    }

    std::cout << StringPrintf(" %d in %.3fs (connected %d)", num_images,
                              timer.ElapsedSeconds(),
                              connected_image_ids.size())
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build correspondence graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Building correspondence graph..." << std::flush;

  for (const auto& image : images_) {
    correspondence_graph_.AddImage(image.first, image.second.NumPoints2D());
  }

  size_t num_ignored_image_pairs = 0;
  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    if (UseInlierMatchesCheck(two_view_geometries[i])) {
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
      if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
        correspondence_graph_.AddCorrespondences(
            image_id1, image_id2, two_view_geometries[i].inlier_matches);
      } else {
        num_ignored_image_pairs += 1;
      }
    } else {
      num_ignored_image_pairs += 1;
    }
  }

  correspondence_graph_.Finalize();

  // Set number of observations and correspondences per image.
  for (auto& image : images_) {
    image.second.SetNumObservations(
        correspondence_graph_.NumObservationsForImage(image.first));
    image.second.SetNumCorrespondences(
        correspondence_graph_.NumCorrespondencesForImage(image.first));
  }

  std::cout << StringPrintf(" in %.3fs (ignored %d)", timer.ElapsedSeconds(),
                            num_ignored_image_pairs)
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Build view graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Building view graph..." << std::flush;

  // 遍历所有匹配上的图像对, 创建view_graph
  num_ignored_image_pairs = 0;
  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    if (UseInlierMatchesCheck(two_view_geometries[i])) {
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
      TwoViewGeometry& two_view_geometry =
          two_view_geometries.at(map_image_pair_ids.at(image_pair_ids[i]));
      class Image image1 = Image(image_id1);
      class Image image2 = Image(image_id2);
      class Camera camera1 = Camera(image1.CameraId());
      class Camera camera2 = Camera(image2.CameraId());
      std::vector<Eigen::Vector2d> points1;
      points1.reserve(image1.NumPoints2D());
      for (const auto& point : image1.Points2D()) {
        points1.push_back(point.XY());
      }

      std::vector<Eigen::Vector2d> points2;
      for (const auto& point : image2.Points2D()) {
        points2.push_back(point.XY());
      }

      // From image1 to image2 Tc2c1
      if (two_view_geometry.EstimateRelativePose(camera1, points1, camera2,
                                                 points2)) {
        // Add the mathcing pair into view graph
        view_graph_.AddEdge(image_id1, image_id2, two_view_geometry);
      } else {
        num_ignored_image_pairs += 1;
      }
    } else {
      num_ignored_image_pairs += 1;
    }
  }

  std::cout << StringPrintf(" in %.3fs (ignored %d)", timer.ElapsedSeconds(),
                            num_ignored_image_pairs)
            << std::endl;
}

void DatabaseCache::Load(const Database& database,
                         const std::string& image_path,
                         const std::string& depth_path,
                         const std::string& image_json_file,
                         const size_t min_num_matches,
                         const bool ignore_watermarks,
                         const bool arkit_view_graph,
                         const std::unordered_set<std::string>& image_names) {
  //////////////////////////////////////////////////////////////////////////////
  // Load cameras
  //////////////////////////////////////////////////////////////////////////////

  Timer timer;

  timer.Start();
  std::cout << "Loading cameras..." << std::flush;

  {
    std::vector<class Camera> cameras = database.ReadAllCameras();
    cameras_.reserve(cameras.size());
    for (auto& camera : cameras) {
      const camera_t camera_id = camera.CameraId();
      cameras_.emplace(camera_id, std::move(camera));
    }
  }

  std::cout << StringPrintf(" %d in %.3fs", cameras_.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Load matches
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading matches..." << std::flush;

  std::vector<image_pair_t> image_pair_ids;
  std::vector<TwoViewGeometry> two_view_geometries;
  std::unordered_map<image_pair_t, int> map_image_pair_ids;
  database.ReadTwoViewGeometries(&image_pair_ids, &two_view_geometries,
                                 &map_image_pair_ids);

  std::cout << StringPrintf(" %d in %.3fs", image_pair_ids.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  auto UseInlierMatchesCheck = [min_num_matches, ignore_watermarks](
                                   const TwoViewGeometry& two_view_geometry) {
    return static_cast<size_t>(two_view_geometry.inlier_matches.size()) >=
               min_num_matches &&
           (!ignore_watermarks ||
            two_view_geometry.config != TwoViewGeometry::WATERMARK);
  };

  //////////////////////////////////////////////////////////////////////////////
  // Load images
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Loading images..." << std::flush;

  std::unordered_set<image_t> image_ids;
  std::unordered_map<std::string, image_t> image_name_to_ids;

  {
    std::vector<class Image> images = database.ReadAllImages();
    const size_t num_images = images.size();

    // Determines for which images data should be loaded.
    if (image_names.empty()) {
      for (const auto& image : images) {
        image_ids.insert(image.ImageId());
        image_name_to_ids.insert(std::make_pair(image.Name(), image.ImageId()));
      }
    } else {
      for (const auto& image : images) {
        if (image_names.count(image.Name()) > 0) {
          image_ids.insert(image.ImageId());
          image_name_to_ids.insert(
              std::make_pair(image.Name(), image.ImageId()));
        }
      }
    }

    // Collect all images that are connected in the correspondence graph.
    std::unordered_set<image_t> connected_image_ids;
    connected_image_ids.reserve(image_ids.size());
    for (size_t i = 0; i < image_pair_ids.size(); ++i) {
      if (UseInlierMatchesCheck(two_view_geometries[i])) {
        image_t image_id1;
        image_t image_id2;
        Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
        if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
          connected_image_ids.insert(image_id1);
          connected_image_ids.insert(image_id2);
        }
      }
    }

    // Load images with correspondences and discard images without
    // correspondences, as those images are useless for SfM.
    images_.reserve(connected_image_ids.size());
    for (auto& image : images) {
      const image_t image_id = image.ImageId();
      if (image_ids.count(image_id) > 0 &&
          connected_image_ids.count(image_id) > 0) {
        images_.emplace(image_id, std::move(image));
        const auto keypoints_and_depths =
            database.ReadKeypointsAndDepths(image_id);
        const auto point2ds = FeatureKeypointsAndDepthsToPoint2Ds(
            keypoints_and_depths.first, keypoints_and_depths.second);
        images_[image_id].SetPoints2D(point2ds);
      }
    }

    std::cout << StringPrintf(" %d in %.3fs (connected %d)", num_images,
                              timer.ElapsedSeconds(),
                              connected_image_ids.size())
              << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Build correspondence graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Building correspondence graph..." << std::flush;

  for (const auto& image : images_) {
    correspondence_graph_.AddImage(image.first, image.second.NumPoints2D());
  }

  size_t num_ignored_image_pairs = 0;
  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    if (UseInlierMatchesCheck(two_view_geometries[i])) {
      image_t image_id1;
      image_t image_id2;
      Database::PairIdToImagePair(image_pair_ids[i], &image_id1, &image_id2);
      if (image_ids.count(image_id1) > 0 && image_ids.count(image_id2) > 0) {
        correspondence_graph_.AddCorrespondences(
            image_id1, image_id2, two_view_geometries[i].inlier_matches);
      } else {
        num_ignored_image_pairs += 1;
      }
    } else {
      num_ignored_image_pairs += 1;
    }
  }

  correspondence_graph_.Finalize();

  // Set number of observations and correspondences per image.
  for (auto& image : images_) {
    image.second.SetNumObservations(
        correspondence_graph_.NumObservationsForImage(image.first));
    image.second.SetNumCorrespondences(
        correspondence_graph_.NumCorrespondencesForImage(image.first));
  }

  std::cout << StringPrintf(" in %.3fs (ignored %d)", timer.ElapsedSeconds(),
                            num_ignored_image_pairs)
            << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Build view graph
  //////////////////////////////////////////////////////////////////////////////

  timer.Restart();
  std::cout << "Building view graph..." << std::flush;

  if (!ExistsFile(image_json_file)) {
    std::cout << "ERROR: image json file doesn't exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::ifstream file(image_json_file);
  if (!file.is_open()) {
    throw std::runtime_error(StringPrintf("Can't open image json file: %s",
                                          image_json_file.c_str()));
  }

  std::vector<ArkitPose> arkit_poses;
  std::unordered_map<std::string, int> image_name_to_arkit_ids;
  json data = json::parse(file);
  auto& elements = data.at("elements");
  for (auto iter = elements.begin(); iter != elements.end(); ++iter) {
    ArkitPose arkit_pose;
    arkit_pose.time_stamp = iter->at("timeStamp");
    arkit_pose.image_name = iter->at("image");
    arkit_pose.depth_name = iter->at("depth");
    arkit_pose.height = iter->at("height");
    arkit_pose.width = iter->at("width");
    arkit_pose.tof_height = iter->at("tofHeight");
    arkit_pose.tof_width = iter->at("tofWidth");
    auto& rotation = iter->at("rotation");
    auto& position = iter->at("position");
    arkit_pose.rotation[0] = rotation.at("w");
    arkit_pose.rotation[1] = rotation.at("x");
    arkit_pose.rotation[2] = rotation.at("y");
    arkit_pose.rotation[3] = rotation.at("z");
    arkit_pose.position[0] = position.at("x");
    arkit_pose.position[1] = position.at("y");
    arkit_pose.position[2] = position.at("z");
    image_name_to_arkit_ids[arkit_pose.image_name] = arkit_poses.size();
    arkit_poses.emplace_back(arkit_pose);
    arkit_poses_[arkit_pose.image_name].qvec = arkit_pose.rotation;
    arkit_poses_[arkit_pose.image_name].tvec = arkit_pose.position;
  }

  num_ignored_image_pairs = 0;

  for (size_t i = 0; i < image_pair_ids.size(); ++i) {
    if (!UseInlierMatchesCheck(two_view_geometries[i])) {
      num_ignored_image_pairs++;
      continue;
    }

    image_pair_t pair_id = image_pair_ids[i];
    image_t image_id1;
    image_t image_id2;
    Database::PairIdToImagePair(pair_id, &image_id1, &image_id2);
    class Image& image1 = Image(image_id1);
    class Image& image2 = Image(image_id2);
    class Camera& camera1 = Camera(image1.CameraId());
    class Camera& camera2 = Camera(image2.CameraId());
    if (!camera1.HasPriorFocalLength() || !camera2.HasPriorFocalLength()) {
      continue;
    }

    const auto arkit_idx1 = image_name_to_arkit_ids.at(image1.Name());
    const auto arkit_idx2 = image_name_to_arkit_ids.at(image2.Name());
    const auto& image_1_arkit = arkit_poses.at(arkit_idx1);
    const auto& image_2_arkit = arkit_poses.at(arkit_idx2);

    const auto& image_1_path = JoinPaths(image_path, image1.Name());
    const auto& image_2_path = JoinPaths(image_path, image2.Name());
    const auto& depth_1_path = JoinPaths(depth_path, image_1_arkit.depth_name);
    const auto& depth_2_path = JoinPaths(depth_path, image_2_arkit.depth_name);

    image1.SetTimeStamp(image_1_arkit.time_stamp); 
    image2.SetTimeStamp(image_2_arkit.time_stamp);
    image1.SetRGBPath(image_1_path);
    image1.SetDepthPath(depth_1_path);
    image2.SetRGBPath(image_2_path);
    image2.SetDepthPath(depth_2_path);

    if (arkit_view_graph) {
      TwoViewGeometry two_view_geometry;
      two_view_geometry.config = TwoViewGeometry::ConfigurationType::ARKIT;
      ComputeRelativePoseArkit(image_1_arkit.rotation, image_1_arkit.position,
                               image_2_arkit.rotation, image_2_arkit.position,
                               &two_view_geometry.qvec,
                               &two_view_geometry.tvec);
      view_graph_.AddEdge(image_id1, image_id2, two_view_geometry);
    } else {
      TwoViewGeometry two_view_geometry = two_view_geometries[i];
      two_view_geometry.config = TwoViewGeometry::ConfigurationType::RGB;
      view_graph_.AddEdge(image_id1, image_id2, two_view_geometry);
    }
  }

  std::cout << StringPrintf(" in %.3fs (edge %d, ignored %d)",
                            timer.ElapsedSeconds(), view_graph_.NumEdges(),
                            num_ignored_image_pairs)
            << std::endl;
}

const class Image* DatabaseCache::FindImageWithName(
    const std::string& name) const {
  for (const auto& image : images_) {
    if (image.second.Name() == name) {
      return &image.second;
    }
  }
  return nullptr;
}

}  // namespace colmap
