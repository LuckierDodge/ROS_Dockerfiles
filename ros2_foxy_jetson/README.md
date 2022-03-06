# ROS 2 Foxy Desktop

Container that bundles ROS 2's Foxy release, with the full desktop install. This version is customized for running on an Nvidia Jetson Nano board.

## Steps to use:

1. Edit `build.sh`, and update any variables to match desired configuration.
1. Run `./build.sh`
1. Edit `launch.sh`, and update any variables to match desired configuration.
1. Do any pre-launch steps, such as starting your X-server on Windows.
1. Run `./launch.sh` to start the docker container. You should find yourself in a docker container with ROS2 Foxy installed. Run `ros2 doctor` to make sure everything is configured properly
1. If you want to attach to a running `ros2_foxy` container, simply run `./attach.sh` from another terminal
1. Note: whenever you make changes to the configuration in one script, be sure to update the other two to reflect the same changes
