cmake_minimum_required(VERSION 3.5)
project(eqr1_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY  meshes  urdf terrains
  DESTINATION share/${PROJECT_NAME}
)

# install(
#   FILES ehr02_description.rviz
#   DESTINATION share/ehr021_description
# )

#ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.dsv.in")

ament_package()


