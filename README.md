# Model-pool ⛵
- It is a collections of all the models that I have created or fine-tuned.
- You can access the weight file by clicking [⚖🏋here🏋⚖](https://drive.google.com/drive/folders/1-IVdOHjcVkgDws0A3LLIV9gZg2oBSAlU?usp=sharing)
- Right now it is in incomplete stage but stay tune to more of it 😁

# How to use it ? 🤔
- Download and install appropriate version of [git](https://git-scm.com/downloads) and [python 3.7+](https://www.python.org/downloads/release/python-378/) in your pc 
- Open a terminal and type in the below command to install **virtualenv** module
~~~

pip install virtualenv

~~~
- Create a new folder and use `Shift + rightclick` to open up a powershell/command prompt (if you are in windows)
- Type in the below command to clone this repository
~~~

git clone https://github.com/HOTSONHONET/Model-pool.git
cd Model-pool.git

~~~
- Copy and paste this set of commands to create a virtual environment and install all the dependecies from **requirements.txt**
~~~

virtualenv ./venv
../venv/Scripts/activate
pip install -r requirements.txt

~~~
- [Download 🔽](https://drive.google.com/drive/folders/1-IVdOHjcVkgDws0A3LLIV9gZg2oBSAlU?usp=sharing) the weights/Resources🏋🏼‍♀️ folder and add it to the [src](https://github.com/HOTSONHONET/Model-pool/tree/main/src) directory.
- The *src* directory will look something like this
![image](https://user-images.githubusercontent.com/56304060/135649971-bbf1b3f7-315b-4956-b9d7-b787fa29cb41.png)
- Now, you are all set and ready to use the models. You can check out models using the below command
~~~

python main.py

~~~
- This will invoke the [main.py](https://github.com/HOTSONHONET/Model-pool/blob/main/main.py) script where you can cycle through the models.  
- I have also added some sample images and video which you can use to test the models and feel free to use your personal data 😉

# Folder structure 📚
- Parent Folder structure

![image](https://user-images.githubusercontent.com/56304060/135654086-3f14feba-d4e5-4565-ad91-85850f99629e.png)
- Model-Pool structure
```
MODEL-POOL
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
│
├───Output folder
│   │   emptyFile
│   │
│   └───collectImages
│           emptyFile
│
├───src
│   │   maskNonMask.py
│   │   skinCancer.py
│   │   yolo_object_detect.py
│   │   __init__.py
│   │
│   └───Resources
│       │   XceptionNN.h5
│       │
│       ├───face_detector
│       │       deploy.prototxt
│       │       res10_300x300_ssd_iter_140000.caffemodel
│       │
│       ├───MaskedFacesClassifier
│       │   │   saved_model.pb
│       │   │
│       │   ├───assets
│       │   └───variables
│       │           variables.data-00000-of-00001
│       │           variables.index
│       │
│       └───yolo
│               coco.names
│               yolov3.cfg
│               yolov3.weights
│
└───Testing images
    ├───Mask and nonMask
    │       video.mp4
    │
    ├───Sample images for yolo
    │       2.jpg
    │       boring.jpg
    │       panda.jpg
    │       people.jpg
    │       traffic.jpg
    │
    └───SkinCancer images
            1.jpg
            2.png
            Actinic keratoses.jpg
            basal_cell_carcinoma_b_high.jpg
            Benign keratosis-like lesions.jpg
            dermatofibroma.jpg
            Melanocytic nevi.jpg
            Melanoma.jpg
            Vascular lesions.png
```
# Sample Images 📸😉
<table>
  <tr>
    <td>Skin Cancer Classification</td>
    <td>Face Mask Detection on Video</td>
    <td>Yolo Object Detection</td>
  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/56304060/135660421-f14906be-1a98-4082-910c-5aecb55d476c.gif" width=500 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/56304060/135671478-dcf2ff0d-20ae-44dd-b230-dbf2bd7d8141.gif" width=500 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/56304060/135658725-315c6311-d175-4db2-ba8b-8476a205ac9a.gif" width=500 height=200></td>
  </tr>
 </table>

