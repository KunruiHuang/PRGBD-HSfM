#ifndef COLMAP_KEYFRAME_EXTRACTOR_H_
#define COLMAP_KEYFRAME_EXTRACTOR_H_

#include <string>

#include "util/math.h"
#include "base/common.h"
#include "util/threading.h"

namespace colmap {

struct KeyFrameExtractorOptions {
   std::string input_path;
   std::string output_path;
  double rot_change = 5.0;
  double pos_change = 0.1;

  bool Check() const;
};

class KeyFrameExtractor : public Thread {
 public:
  KeyFrameExtractor(const KeyFrameExtractorOptions& options);

 private:

  void Run() override;

  bool ExtractorKeyFrames() const;

  KeyFrameExtractorOptions options_;
};

} // namespace colmap


#endif  // COLMAP_KEYFRAME_EXTRACTOR_H_
