FROM ros:humble-ros-core

# Update GPG that is outdated in the ros:humble-ros-core image.
RUN rm /etc/apt/sources.list.d/ros*
RUN apt update && apt install curl -y --no-install-recommends
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install packages and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-colcon-common-extensions \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

## Install python dependencies
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install roma

# Create workspace
RUN apt-get update && apt-get install -y git
RUN mkdir workspace
WORKDIR /workspace

# COPY ros_entrypoint.sh /ros_entrypoint.sh

# WORKDIR /colcon_ws

# COPY ./goat_shape_state_estimation src/goat_shape_state_estimation 

# RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
#     colcon build --symlink-install --event-handlers console_direct+ --cmake-args ' -DCMAKE_BUILD_TYPE=Release'

# # Set package's launch command
# ENV LAUNCH_COMMAND='ros2 run goat_shape_state_estimation goat_shape_state_estimation'

# # Create build and run aliases
# RUN echo 'alias build="colcon build --symlink-install  --event-handlers console_direct+"' >> /etc/bash.bashrc && \
#     echo 'alias run="su - ros --whitelist-environment=\"ROS_DOMAIN_ID\" /run.sh"' >> /etc/bash.bashrc && \
#     echo "source /colcon_ws/install/setup.bash; echo UID: $UID; echo ROS_DOMAIN_ID: $ROS_DOMAIN_ID; $LAUNCH_COMMAND" >> /run.sh && chmod +x /run.sh