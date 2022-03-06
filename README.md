# ROS_Dockerfiles

This repository contains a collection of example Dockerfiles and scripts designed to help robotics developers create and share portable development environments.

Within each directory is a README.md with any instructions for how to build, run, and configure each container, as well as special notes for that particular build.
The containers are sandboxed, with their own programs, operating system, and filestystems, but in many cases they have volumes from the host filesystem mounted into them to support development.
In addition, many may have access to the host's network, graphics, and devices.
These accesses will be noted in the README.md as well.

## Notes for WSL2

If you intend to use these scripts on Windows, I recommend using WSL2.

1. Install a WSL2 distro, then install Docker inside. [You can follow this guide](https://dev.to/luckierdodge/how-to-install-and-use-docker-in-wsl2-217l).
1. To take advantage of the graphical user interface features of these containers, do the following:
    1. Run `sudo apt install x11-xserver-utils`
    1. In Windows, install an X-server, like [VcXsrv](https://sourceforge.net/projects/vcxsrv/). See [here](https://teamdynamix.umich.edu/TDClient/47/LSAPortal/KB/ArticleDet?ID=1797) for some other options.
    1. Before running `launch.sh` or otherwise starting the container, start your X-server (For VcXSrv, run the command "XLaunch", then choose the following options: Multiple windows, Display Number -1, "Start no Client", "Disable access control").
    1. In WSL, run the following before running `launch.sh`: `export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0.0` (you may add this to your `~/.profile` or `~/.bashrc`, if you want it to run automatically)