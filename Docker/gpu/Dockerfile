FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# ENV ROS_DISTRO kinetic
ENV ROS_DISTRO melodic
###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=po \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

RUN apt-get update && apt-get install -y \
    curl \
    lsb-release \
    sudo \
    software-properties-common \
    git \
    cmake \
    make \
    vim \
    build-essential \
    libblkid-dev \
    e2fslibs-dev \
    libboost-all-dev \
    libaudit-dev \
    tzdata \
    wget \
    openssh-server

###################################### python version update #####################################

RUN add-apt-repository ppa:deadsnakes/ppa && apt update && apt install python3.6 -y
RUN apt-get remove python-pip -y
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN wget https://bootstrap.pypa.io/pip/2.7/get-pip.py
RUN python2.7 get-pip.py.1
RUN hash -r

###################################### ros install（kinetic） #####################################

# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt install curl
# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
# RUN apt-get update && apt-get install ros-kinetic-desktop-full -y

###################################### ros install（melodic） #####################################

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
RUN apt update && apt install ros-melodic-desktop -y

###################################### REALSENSE #####################################

RUN apt-get update && apt-get install -y software-properties-common
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
RUN sudo apt-get install -y librealsense2-dkms
RUN apt-get install -y librealsense2-utils
RUN apt-get install -y librealsense2-dev
RUN apt-get install -y librealsense2-dbg
RUN apt-get install -y v4l-utils

###################################### Dockerize an SSH service #####################################

RUN mkdir /var/run/sshd
RUN echo 'root:screencast' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
  
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
  
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
  
EXPOSE 22

# #################################### GTSAM ####################################

# RUN cd ${HOME}/ \
#     && git clone https://github.com/borglab/gtsam.git \
#     && cd gtsam/ \
#     && mkdir build \
#     && cd build \
#     && cmake ..  \
#     && make -j12 \
#     && make install
# ############################# ROS pkg ############################

RUN apt-get update && apt-get install -y \
    ros-melodic-vision-visp \
    ros-melodic-hardware-interface \
    ros-melodic-pcl-ros \
    ros-melodic-moveit \
    ros-melodic-gazebo-ros-pkgs \
    ros-melodic-ros-control

# ############################# LoCoBot and PyRobot ############################
# RUN python -m pip install -U pip setuptools

# RUN cd ${HOME}/

# RUN curl 'https://raw.githubusercontent.com/jack6099boy/pyrobot/master/robots/LoCoBot/install/locobot_install_all.sh' > locobot_install_all.sh && chmod +x locobot_install_all.sh

# RUN ./locobot_install_all.sh -t sim_only -p 2 -l interbotix

# RUN rm locobot_install_all.sh

# RUN apt-get update && apt-get install -y \
#     ros-melodic-smach \
#     ros-melodic-smach-msgs \
#     ros-melodic-smach-ros \
#     ros-melodic-executive-smach \
#     ros-melodic-smach-viewer

# ##################################### UNSEENsegment object ######################################

RUN pip2 install torch future scipy torchvision easydict transforms3d

# ##################################### Pytorch ######################################
# RUN python3 -m pip install -U pip 

# RUN pip3 install empy \
#     catkin-tools \
#     opencv-python \ 
#     rospkg \
#     matplotlib \
#     wget 

# RUN python3 -m pip install \
#     torch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html


############################# custom gazebo ###############################
# pre download model 
# RUN mkdir -p $HOME/.gazebo \
#     && cd $HOME/.gazebo \
#     && git clone https://github.com/osrf/gazebo_models \
#     && mv gazebo_models models \
#     && cd


# RUN python -m pip install future
# RUN /usr/bin/python -m pip install --upgrade pip
# RUN python -m pip install open3d
# RUN apt install -y ros-melodic-ros-numpy
# RUN python -m pip install pyrsistent==0.13 open3d_ros_helper

# RUN python -m pip install install torch torchvision 
# RUN python -c "import torchvision; torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)"

# RUN python3 -m pip install gshell
# RUN apt-get install -y ros-melodic-pointcloud-to-laserscan
# RUN apt-get install -y ros-melodic-people
##################################### setting #####################################
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL) ALL" > /etc/sudoers

RUN echo "root ALL=(ALL)  ALL" > /etc/sudoers

RUN echo "cd ~/TM5" >> ${HOME}/.bashrc

# GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# setup entrypoint
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]

USER ${USER}
WORKDIR ${HOME}
# RUN gdown https://drive.google.com/uc?id=1dEzygT9CTtQDZU2xgjWO4qvVjyG2-kjj


# RUN ls
CMD ["/usr/sbin/sshd", "-D"]