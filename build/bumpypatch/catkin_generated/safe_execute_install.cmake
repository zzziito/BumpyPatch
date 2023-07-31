execute_process(COMMAND "/home/rtlink/jiwon/paper_ws/build/bumpypatch/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/rtlink/jiwon/paper_ws/build/bumpypatch/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
