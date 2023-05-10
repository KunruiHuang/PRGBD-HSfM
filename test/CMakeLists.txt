set(FOLDER_NAME "test")

if(IS_MSVC)
add_compile_options("/bigobj")
endif()

COLMAP_ADD_EXECUTABLE(test_exe
test.cc
  test_global_rotation.h
  test_global_rotation.cc
)
set_target_properties(test_exe PROPERTIES OUTPUT_NAME test)

