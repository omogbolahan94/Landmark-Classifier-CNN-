## Project Overview

Welcome to the Convolutional Neural Networks (CNN) project!
In this project, you will learn how to build a pipeline to process real-world, user-supplied images and to put your model into an app.
Given an image, your app will predict the most likely locations where the image was taken.

By completing this lab, you demonstrate your understanding of the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. 

Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.

### Why We're Here

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernable landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

In this project, I took the first steps towards addressing this problem by building a CNN-powered app to automatically predict the location of the image based on any landmarks depicted in the image. At the end of this project, the app will accept any user-supplied image as input and suggest the top k (e.g top 5) most relevant landmarks from 50 possible landmarks from across the world.


## Project Instructions

### Getting started

I tranined the convolutional neural network in the provided Udacity GPU enabled workspace in my classroom. I also utilised ResNet-18 pre-trained in model 

#### Setting up locally

This setup requires a bit of familiarity with creating a working deep learning environment. While things should work out of the box, in case of problems you might have to do operations on your system (like installing new NVIDIA drivers). Please do this if you are at least a bit familiar with these subjects, otherwise please consider using an hosted GPU on the cloud.

1. Open a terminal and clone the repository, then navigate to the downloaded folder:
	
	```	
		git clone https://github.com/omogbolahan94/Landmark-Classifier-CNN-.git
		cd Landmark-Classifier-CNN-
	```
    
2. Create a new conda environment with python 3.7.6:

    ```
        conda create --name cnn_project -y python=3.7.6
        conda activate cnn_project
    ```
    
    NOTE: you will have to execute `conda activate cnn_project` for every new terminal session.
    
3. Install the requirements of the project:

    ```
        pip install -r requirements.txt
    ```

4. Install and open Jupyter lab:
	
	```
        pip install jupyterlab
		jupyter lab
	```

### Developing your project

With the Udacity working environment (GPU environment), I executed the following steps in order:

1. Opened the `cnn_from_scratch.ipynb` notebook and follow the instructions there
2. Opened `transfer_learning.ipynb` and follow the instructions
3. Opened `app.ipynb` and follow the instructions there

### Evaluation

This project was reviewed by a professional Udacity reviewer against the CNN project rubric. I reviewed the rubric thoroughly and self-evaluate my project before submission. All criteria found in the rubric met specifications that requires my passing of the project.

### Dataset Info

The landmark images are a subset of the Google Landmarks Dataset v2.
