FROM tensorflow/tensorflow:latest-gpu
RUN apt update && apt upgrade -yqq
RUN apt-get install ffmpeg libsm6 libxext6  -yqq
RUN pip3 install -U jupyter pandas opencv-python matplotlib scikit-learn
CMD jupyter notebook --ip 0.0.0.0 --allow-root