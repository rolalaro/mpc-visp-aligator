cmake_minimum_required(VERSION 3.28)
project(mpc-example)

find_package (aligator REQUIRED)
find_package(pinocchio REQUIRED)
find_package(proxsuite REQUIRED)
find_package(VISP REQUIRED)

add_executable(mpc-example mpc-example.cpp)
target_link_libraries(mpc-example PUBLIC aligator::aligator pinocchio::pinocchio)

add_executable(viper850-mpc-example viper850-mpc-example.cpp)
target_link_libraries(viper850-mpc-example PUBLIC ${VISP_LIBRARIES} aligator::aligator pinocchio::pinocchio)
target_include_directories(viper850-mpc-example PUBLIC ${VISP_INCLUDE_DIRS})

add_executable(viper850-qp-example viper850-qp-example.cpp)
target_link_libraries(viper850-qp-example PUBLIC ${VISP_LIBRARIES} proxsuite::proxsuite pinocchio::pinocchio)
target_include_directories(viper850-qp-example PUBLIC ${VISP_INCLUDE_DIRS})

set(URDF_FOLDER ${CMAKE_BINARY_DIR}/urdf)

add_custom_command(
  OUTPUT ${URDF_FOLDER}/viper850_sim.urdf
  COMMAND ${CMAKE_COMMAND} -E make_directory ${URDF_FOLDER}
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/urdf/viper850_sim.urdf ${URDF_FOLDER}/viper850_sim.urdf
  DEPENDS viper850-mpc-example
  COMMENT "Copying URDF for the example"
  VERBATIM
)

add_custom_target(
  copy_urdf ALL
  DEPENDS ${URDF_FOLDER}/viper850_sim.urdf
  VERBATIM
)
