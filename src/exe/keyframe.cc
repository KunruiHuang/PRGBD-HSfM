
#include "exe/keyframe.h"

#include <string>
#include <iostream>
#include <memory>

#include "util/misc.h"
#include "base/keyframe_extractor.h"
#include "util/option_manager.h"

namespace colmap {

int RunKeyFrameExtractor(int argc, char** argv) {
  OptionManager options;
  options.AddKeyFrameExtractorOptions();
  options.Parse(argc, argv);

  const auto& keyframe_options = *options.keyframe_extraction.get();
  if (!ExistsDir(keyframe_options.input_path)) {
    std::cout << "ERROR: Input path doesn't exist." << std::endl;
    return EXIT_FAILURE;
  }

  if (!ExistsDir(keyframe_options.output_path)) {
    std::cout << "ERROR: Output path doesn't exist." << std::endl;
    return EXIT_FAILURE;
  }

  KeyFrameExtractor key_frame_extractor(keyframe_options);

  key_frame_extractor.Start();
  key_frame_extractor.Wait();

  return EXIT_SUCCESS;
}

} // namespace colmap
