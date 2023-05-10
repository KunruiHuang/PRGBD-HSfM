#include "pose.h"

#include "base/reconstruction.h"
#include <nlohmann/json.hpp>
#include <unordered_set>

#include "base/pose.h"
#include "base/image.h"
#include "util/csv.h"
#include "util/misc.h"
#include "base/pose.h"
#include "util/option_manager.h"

namespace colmap {

using json = nlohmann::json;

int RunConvertArkitPose(int argc, char** argv) {
  std::string sparse_path;
  std::string image_json;
  std::string output_path;
  OptionManager options;
  options.AddRequiredOption("sparse_path", &sparse_path,
                            "reconstruction sparse path");
  options.AddRequiredOption("image_json", &image_json, "image_json file");
  options.AddRequiredOption("output_path", &output_path, "output_path");
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.ReadBinary(sparse_path);

  std::vector<ArkitPose> arkit_poses; 
  std::ifstream file(image_json);
  if (!file.is_open()) {
    throw std::runtime_error(StringPrintf("Can't open image json file: %s",
                                          image_json.c_str()));
  }

  json data = json::parse(file);
  std::unordered_map<std::string, int> image_names_to_ids; 
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
     
    image_names_to_ids[arkit_pose.image_name] = arkit_poses.size();  
    arkit_poses.emplace_back(arkit_pose);
  }

  for (auto& image : reconstruction.Images()) {
    const int id = image_names_to_ids.at(image.second.Name()); 
    const auto& arkit_pose = arkit_poses.at(id);
    Image& image_modified = reconstruction.Image(image.first);
    InvertPose(arkit_pose.rotation, arkit_pose.position, &image_modified.Qvec(), &image_modified.Tvec());
    // image_modified.Qvec() = arkit_pose.rotation; 
    // image_modified.Tvec() = arkit_pose.position; 

  }

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

int RunConvertGlobalRotation(int argc, char** argv) {
  std::string sparse_path;
  std::string global_rotation;
  std::string output_path;
  OptionManager options;
  options.AddRequiredOption("sparse_path", &sparse_path,
                            "reconstruction sparse path");
  options.AddRequiredOption("global_rotation", &global_rotation, "global_rotation file");
  options.AddRequiredOption("output_path", &output_path, "output_path");
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.ReadBinary(sparse_path);

   std::ifstream file(global_rotation, std::ios::binary); 
   CHECK(file.is_open()) << global_rotation; 

   std::unordered_map<image_t, Eigen::Vector4d> global_orientations;

   const size_t num_global_rotations = ReadBinaryLittleEndian<uint64_t>(&file); 
   for (size_t i = 0; i < num_global_rotations; ++i) {
    image_t image_id; 
    Eigen::Vector3d rotation; 
    image_id = ReadBinaryLittleEndian<image_t>(&file);
    rotation[0] = ReadBinaryLittleEndian<double>(&file); 
    rotation[1] = ReadBinaryLittleEndian<double>(&file); 
    rotation[2] = ReadBinaryLittleEndian<double>(&file);
    global_orientations[image_id] = VectorToQuaternion(rotation);
   }

   for (auto& image : reconstruction.Images()) {
    Image& image_modified = reconstruction.Image(image.first);
    image_modified.Qvec() = global_orientations.at(image.first); 
  }

  reconstruction.Write(output_path);

  return EXIT_SUCCESS;
}

}  // namespace colmap
