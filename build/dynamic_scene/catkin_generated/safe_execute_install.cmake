execute_process(COMMAND "/home/rtlink/jiwon/bumpypatch_ws/build/dynamic_scene/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/rtlink/jiwon/bumpypatch_ws/build/dynamic_scene/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
