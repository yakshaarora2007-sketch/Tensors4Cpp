# CMake generated Testfile for 
# Source directory: E:/New folder
# Build directory: E:/New folder/build-asan
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(minitensor_tests "E:/New folder/build-asan/Debug/test_tensor.exe")
  set_tests_properties(minitensor_tests PROPERTIES  _BACKTRACE_TRIPLES "E:/New folder/CMakeLists.txt;81;add_test;E:/New folder/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(minitensor_tests "E:/New folder/build-asan/Release/test_tensor.exe")
  set_tests_properties(minitensor_tests PROPERTIES  _BACKTRACE_TRIPLES "E:/New folder/CMakeLists.txt;81;add_test;E:/New folder/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(minitensor_tests "E:/New folder/build-asan/MinSizeRel/test_tensor.exe")
  set_tests_properties(minitensor_tests PROPERTIES  _BACKTRACE_TRIPLES "E:/New folder/CMakeLists.txt;81;add_test;E:/New folder/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(minitensor_tests "E:/New folder/build-asan/RelWithDebInfo/test_tensor.exe")
  set_tests_properties(minitensor_tests PROPERTIES  _BACKTRACE_TRIPLES "E:/New folder/CMakeLists.txt;81;add_test;E:/New folder/CMakeLists.txt;0;")
else()
  add_test(minitensor_tests NOT_AVAILABLE)
endif()
