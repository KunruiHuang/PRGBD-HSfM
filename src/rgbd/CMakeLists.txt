set(FOLDER_NAME "rgbd")

COLMAP_ADD_SOURCES(
   camera_model.h camera_model.cc
   dataset_rgbd.h dataset_rgbd.cc
   frame.h frame.cc
   lie_algebra.h
   math.h math.cc
   normal_equations.h
   pyramid.h pyramid.cc
   rgbd.h rgbd.cc
   tracker.h tracker.cc
)