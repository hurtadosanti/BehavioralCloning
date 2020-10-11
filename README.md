# Behavioral Cloning
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

This repository contains a solution to the Behavioral Cloning Project.

In this project, we use deep neural networks and convolutional neural networks to clone driving behavior. We train, validate and test a model using Keras. The model outputs a steering angle to an autonomous vehicle.

The goals of this project are the following:

- Use the simulator to collect data of good driving behavior
- Design, train and validate a model that predicts a steering angle from image data
- Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
- Summarize the results with a written report

## Files
The project includes the following files:

* [Report](report.md) summarizing the results
* [Jupyter notebook](BehavioralCloning.ipynb) Describing the steps to load, preprocess and train the model
* Model python [model.py](model.py) executable containing the script to create and train the model
* Driver Server [drive.py](utils/drive.py) for driving the car in autonomous mode
* Trained model [model.h5](model.h5) containing a trained convolution neural network 
* [Video](video.mp4) for an autonomous drive example

Environment
---
- The conda environment can be found in the file [environment.yml](environment.yml)

- The self-driving car simulator built with Unity can be found on this [github repository](https://github.com/udacity/self-driving-car-sim) 

License
---
MIT License Copyright (c) 2016-2018 Udacity, Inc.

References
---
https://github.com/udacity/CarND-Behavioral-Cloning-P3


