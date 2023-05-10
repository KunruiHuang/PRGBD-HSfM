#include "base/keyframe_extractor.h"

#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <unordered_set>

#include "base/pose.h"
#include "util/logging.h"
#include "util/math.h"
#include "util/misc.h"
#include "util/string.h"
#include "util/timer.h"

namespace colmap {
namespace {

using json = nlohmann::json;

void ReadArkitPoses(const std::string& image_json_file,
                    std::vector<ArkitPose>* arkit_poses) {
  CHECK(arkit_poses);
  std::ifstream file(image_json_file);
  if (!file.is_open()) {
    throw std::runtime_error(StringPrintf("Can't open image json file: %s",
                                          image_json_file.c_str()));
  }

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

    auto& intrinsics = iter->at("intrinsics");
    auto& distortions = iter->at("distortions");
    for (auto item = intrinsics.begin(); item != intrinsics.end(); ++item) {
      arkit_pose.intrinsics.emplace_back(*item);
    }

    for (auto item = distortions.begin(); item != distortions.end(); ++item) {
      arkit_pose.distortions.emplace_back(*item);
    }

    arkit_poses->emplace_back(arkit_pose);
  }

  std::sort(arkit_poses->begin(), arkit_poses->end(),
            [&](const ArkitPose& prev, const ArkitPose& curr) -> bool {
              return prev.time_stamp < curr.time_stamp;
            });

  // Transform the arkit pose format into from camera to world
  Eigen::Matrix3d signed_rot_mat;
  signed_rot_mat << 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0;
  Eigen::Matrix3d signed_mat;
  signed_mat << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0;

  Eigen::Matrix3d ref_rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d ref_position = Eigen::Vector3d::Identity();

  for (size_t i = 0; i < arkit_poses->size(); ++i) {
    auto& arkit_pose = arkit_poses->at(i);
    auto rotation_mat = QuaternionToRotationMatrix(arkit_pose.rotation);
    auto position_mat = arkit_pose.position;

    if (i == 0) {
      ref_rotation = rotation_mat;
      ref_position = position_mat;
    }

    rotation_mat = ref_rotation.transpose() * rotation_mat.eval();
    rotation_mat(0, 1) *= -1.0;
    rotation_mat(0, 2) *= -1.0;
    rotation_mat(1, 0) *= -1.0;
    rotation_mat(2, 0) *= -1.0;
    position_mat = ref_rotation.transpose() * (position_mat - ref_position);
    position_mat.y() *= -1.0;
    position_mat.z() *= -1.0;

    arkit_pose.rotation = RotationMatrixToQuaternion(rotation_mat);
    arkit_pose.position = position_mat;
  }
}

void ExtractorKeyFrameFromArkitPose(const std::vector<ArkitPose>& arkit_poses,
                                    const double rot_change,
                                    const double pos_change,
                                    std::vector<ArkitPose>* key_frames) {
  CHECK(key_frames);
  if (arkit_poses.empty()) {
    return;
  }
  Eigen::Matrix3d prev_rotation;
  Eigen::Vector3d prev_position;

  // const double rot_change_rad = DegToRad(rot_change);

  const int arkit_poses_size = static_cast<int>(arkit_poses.size());
  key_frames->reserve(arkit_poses_size);
  for (int i = 0; i < arkit_poses_size; ++i) {
    const auto& arkit_pose = arkit_poses.at(i);
    const auto& curr_rotation = QuaternionToRotationMatrix(arkit_pose.rotation);
    const auto& curr_position = arkit_pose.position;

    if (i == 0) {
      prev_rotation = curr_rotation;
      prev_position = curr_position;
      key_frames->emplace_back(arkit_pose);
      continue;
    }

    Eigen::Matrix3d delta_R = prev_rotation * curr_rotation.inverse();
    Eigen::Vector3d delta_t = prev_position - delta_R * curr_position;
    double rx, ry, rz;
    RotationMatrixToEulerAngles(delta_R, &rx, &ry, &rz);

    if (delta_t.norm() > pos_change || RadToDeg(rx) > rot_change ||
        RadToDeg(ry) > rot_change || RadToDeg(rz) > rot_change) {
      key_frames->emplace_back(arkit_pose);
      prev_rotation = curr_rotation;
      prev_position = curr_position;
    }
  }

  key_frames->shrink_to_fit();
}

void CopyKeyFrames(const std::string& input_path,
                   const std::string& output_path,
                   std::vector<ArkitPose>* key_frames) {
  CHECK(key_frames);
  if (key_frames->empty()) {
    std::cout << "ERRO: No available keyframes!" << std::endl;
    return;
  }
  std::string raw_image_path = JoinPaths(input_path, "image");
  std::string raw_depth_path = JoinPaths(input_path, "depth");
  std::string image_path = JoinPaths(output_path, "image");
  std::string depth_path = JoinPaths(output_path, "depth");
  CreateDirIfNotExists(image_path);
  CreateDirIfNotExists(depth_path);

  if (!ExistsDir(raw_image_path)) {
    throw std::runtime_error(StringPrintf(
        "Input path %s doesn't have images folder.", input_path.c_str()));
  }

  if (!ExistsDir(raw_depth_path)) {
    throw std::runtime_error(StringPrintf(
        "Input path %s doesn't have depth folder.", input_path.c_str()));
  }

  std::vector<std::string> raw_image_lists =
      GetRecursiveFileList(raw_image_path);

  std::unordered_map<std::string, int> image_name_ids;
  for (size_t i = 0; i < key_frames->size(); ++i) {
    const auto& keyframe = key_frames->at(i);
    image_name_ids.emplace(keyframe.image_name, i);
  }

  for (const auto& item : raw_image_lists) {
    const auto image_name = GetPathBaseName(item);
    if (image_name_ids.count(image_name) <= 0) {
      continue;
    }

    const int idx = image_name_ids.at(image_name);
    auto& keyframe = key_frames->at(idx);

    const std::string raw_depth_name =
        JoinPaths(raw_depth_path, keyframe.depth_name + "_depth.png");

    if (!ExistsFile(raw_depth_name)) {
      std::cerr << "ERROR: No valid depth with the image: " << image_name
                << ", You should convert depth data format by using "
                   "scripts/python/read_depth.py"
                << std::endl;
      exit(EXIT_FAILURE);
    }

    const auto base_depth_name = GetPathBaseName(raw_depth_name);

    const auto dst_image_name = JoinPaths(image_path, image_name);
    const auto dst_depth_name = JoinPaths(depth_path, base_depth_name);

    keyframe.image_name = image_name;
    keyframe.depth_name = base_depth_name;

    if (!ExistsFile(dst_image_name)) {
      FileCopy(item, dst_image_name);
    }

    if (!ExistsFile(dst_depth_name)) {
      FileCopy(raw_depth_name, dst_depth_name);
    }
  }
}

void SaveImageJson(const std::string& image_json_file,
                   const std::vector<ArkitPose>& keyframes) {
  json image_json;
  auto& elements = image_json["elements"];

  for (const auto& item : keyframes) {
    json element;
    element["timeStamp"] = item.time_stamp;
    element["image"] = item.image_name;
    element["depth"] = item.depth_name;
    element["height"] = item.height;
    element["width"] = item.width;
    element["tofWidth"] = item.tof_width;
    element["tofHeight"] = item.tof_height;
    element["rotation"]["w"] = item.rotation[0];
    element["rotation"]["x"] = item.rotation[1];
    element["rotation"]["y"] = item.rotation[2];
    element["rotation"]["z"] = item.rotation[3];
    element["position"]["x"] = item.position[0];
    element["position"]["y"] = item.position[1];
    element["position"]["z"] = item.position[2];
    for (const auto& param : item.intrinsics) {
      element["intrinsics"].emplace_back(param);
    }

    for (const auto& param : item.distortions) {
      element["distortions"].emplace_back(param);
    }

    elements.emplace_back(element);
  }

  std::ofstream output_file(image_json_file);
  if (!output_file.is_open()) {
    throw std::runtime_error(
        StringPrintf("Can't open file %s", image_json_file.c_str()));
  }
  output_file << image_json;
  output_file.close();
}

}  // namespace

bool KeyFrameExtractorOptions::Check() const {
  CHECK_GT(rot_change, 0.0);
  CHECK_GT(pos_change, 0.0);

  return true;
}

KeyFrameExtractor::KeyFrameExtractor(const KeyFrameExtractorOptions& options)
    : Thread(), options_(options) {}

void KeyFrameExtractor::Run() {
  while (true) {
    if (IsStopped()) {
      break;
    }

    if (ExtractorKeyFrames()) {
      break;
    } else {
      std::cout << "ERROR: extractor keyframes failed" << std::endl;
      break;
    }
  }
}

bool KeyFrameExtractor::ExtractorKeyFrames() const {
  std::string image_json_file = JoinPaths(options_.input_path, "images.json");
  if (!ExistsFile(image_json_file)) {
    std::cout << "ERROR: Can't find image json file from `input_path`"
              << std::endl;
    return false;
  }

  PrintHeading1("Extractor KeyFrames");

  Timer timer;

  timer.Start();
  std::cout << "Reading image json..." << std::flush;

  std::vector<ArkitPose> arkit_poses;
  ReadArkitPoses(image_json_file, &arkit_poses);

  std::cout << StringPrintf(" %d in %.3fs", arkit_poses.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  std::cout << "Extractor key frames..." << std::flush;
  timer.Restart();

  std::vector<ArkitPose> key_frames;
  ExtractorKeyFrameFromArkitPose(arkit_poses, options_.rot_change,
                                 options_.pos_change, &key_frames);

  std::cout << StringPrintf(" %d in %.3fs", key_frames.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  std::cout << "Copy key frames..." << std::flush;
  timer.Restart();
  CopyKeyFrames(options_.input_path, options_.output_path, &key_frames);

  std::cout << StringPrintf(" %d in %.3fs", key_frames.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  std::cout << "Save image json..." << std::flush;
  timer.Restart();
  std::string key_frames_json = JoinPaths(options_.output_path, "images.json");
  SaveImageJson(key_frames_json, key_frames);

  std::cout << StringPrintf(" %d in %.3fs", key_frames.size(),
                            timer.ElapsedSeconds())
            << std::endl;

  return true;
}

}  // namespace colmap
