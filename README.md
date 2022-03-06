# ROS_Dockerfiles

This repository contains a collection of example Dockerfiles and scripts designed to help robotics developers create and share portable development environments.

Within each directory is a README.md with any instructions for how to build, run, and configure each container, as well as special notes for that particular build. 
The containers are sandboxed, with their own programs, operating system, and filestystems, but in many cases they have volumes from the host filesystem mounted into them to support development. 
In addition, many may have access to the host's network, graphics, and devices. 
These accesses will be noted in the README.md as well.
