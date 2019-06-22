# Sign Language Detection and Classification

BBM416 - Computer Vision Course Project

### Authors
Ayhan YÜNT - 21228965

Sefa BOZKUŞ - 21327733

Barkın BOZ - 21327728



### Project Introduction
# Be My Voice
Image Detection is a very common subject for every area
for computers and smart phones. The most important
think for making digital tools as humanly as possible is
to making them to understand what they see. In this paper
we present a progress report for Turkish Sign Language
Detector. Sign language is used by deaf and hard hearing
people to exchange information between their own
community and with other people. Computer recognition
of sign language deals from sign gesture acquisition and
continues till text/speech generation. Four essential components
in a gesture recognition system are: gesture modeling,
gesture analysis, gesture recognition and gesturebased
application systems.

    
### Prerequisites

This project uses python 3.5 and the PIP following packages:

    OpenCV - Tensorflow - Numpy - Matplotlib

See requirements.txt for versions.

### Install using PIP
```
pip3 install -r requirements.txt
```

### Running the Train

    python train.py [options]
    
    Options:
        -b, --bottleneck-dir : (Location of cached images.)
        -e, --epoch-num : (Number of epoch for training.)
        -m, --model-directory : (Location of downloaded model.)
        -s, --summaries-directory: (Location of training summaries.)
        -o, --output-graph: (Location of output graphics.)
        -l, --output-labels: (Location of predicted labels.)
        -d, --dataset-directory: (Location of dataset.)
        
    Default Options: (If you dont write any options)
        -b logs/bottlenecks
        -e 15000
        -m inception
        -s logs/training_summary/basic
        -o logs/trained_graph.pb
        -l logs/trained_labels.txt
        -d dataset

-   Program uses default parameters when parameters are not specified.

-   Output: Loss, Train Accuracy, Test Accuracy are shown in console and best model will be stored
in disk for using in test.

- Expected result on console: 
```
Step: 0, Train accuracy: 14.0000%, Cross entropy: 3.743247, Validation accuracy: 2.0% (N=100)
Step: 100, Train accuracy: 42.0000%, Cross entropy: 2.962920, Validation accuracy: 50.0% (N=100)
Step: 200, Train accuracy: 72.0000%, Cross entropy: 2.454993, Validation accuracy: 50.0% (N=100)
Step: 300, Train accuracy: 63.0000%, Cross entropy: 2.315188, Validation accuracy: 62.0% (N=100)
Step: 400, Train accuracy: 75.0000%, Cross entropy: 2.002585, Validation accuracy: 71.0% (N=100)
Step: 500, Train accuracy: 72.0000%, Cross entropy: 1.834684, Validation accuracy: 68.0% (N=100)
Step: 600, Train accuracy: 73.0000%, Cross entropy: 1.828362, Validation accuracy: 75.0% (N=100)
Step: 700, Train accuracy: 80.0000%, Cross entropy: 1.556804, Validation accuracy: 75.0% (N=100)
Step: 800, Train accuracy: 84.0000%, Cross entropy: 1.481381, Validation accuracy: 79.0% (N=100)
Step: 900, Train accuracy: 75.0000%, Cross entropy: 1.558882, Validation accuracy: 77.0% (N=100)
Step: 1000, Train accuracy: 89.0000%, Cross entropy: 1.311620, Validation accuracy: 81.0% (N=100)
Step: 1100, Train accuracy: 83.0000%, Cross entropy: 1.310728, Validation accuracy: 76.0% (N=100)
Step: 1200, Train accuracy: 85.0000%, Cross entropy: 1.190230, Validation accuracy: 74.0% (N=100)
Step: 1300, Train accuracy: 81.0000%, Cross entropy: 1.359421, Validation accuracy: 81.0% (N=100)
Step: 1400, Train accuracy: 82.0000%, Cross entropy: 1.175252, Validation accuracy: 82.0% (N=100)
Step: 1500, Train accuracy: 84.0000%, Cross entropy: 1.104121, Validation accuracy: 81.0% (N=100)
Step: 1600, Train accuracy: 89.0000%, Cross entropy: 1.029864, Validation accuracy: 78.0% (N=100)
Step: 1700, Train accuracy: 89.0000%, Cross entropy: 0.928768, Validation accuracy: 85.0% (N=100)
Step: 1800, Train accuracy: 88.0000%, Cross entropy: 1.017034, Validation accuracy: 81.0% (N=100)
Step: 1900, Train accuracy: 86.0000%, Cross entropy: 1.006394, Validation accuracy: 82.0% (N=100)
Step: 2000, Train accuracy: 89.0000%, Cross entropy: 0.865934, Validation accuracy: 86.0% (N=100)
Step: 2100, Train accuracy: 91.0000%, Cross entropy: 0.836143, Validation accuracy: 87.0% (N=100)
Step: 2200, Train accuracy: 88.0000%, Cross entropy: 0.880344, Validation accuracy: 84.0% (N=100)
Step: 2300, Train accuracy: 91.0000%, Cross entropy: 0.855649, Validation accuracy: 86.0% (N=100)
Step: 2400, Train accuracy: 91.0000%, Cross entropy: 0.758076, Validation accuracy: 86.0% (N=100)
Step: 2500, Train accuracy: 92.0000%, Cross entropy: 0.784573, Validation accuracy: 91.0% (N=100)
Step: 2600, Train accuracy: 86.0000%, Cross entropy: 0.938465, Validation accuracy: 86.0% (N=100)
Step: 2700, Train accuracy: 84.0000%, Cross entropy: 0.903375, Validation accuracy: 83.0% (N=100)
Step: 2800, Train accuracy: 88.0000%, Cross entropy: 0.822930, Validation accuracy: 83.0% (N=100)
Step: 2900, Train accuracy: 89.0000%, Cross entropy: 0.742477, Validation accuracy: 89.0% (N=100)
Step: 3000, Train accuracy: 85.0000%, Cross entropy: 0.769462, Validation accuracy: 84.0% (N=100)
Step: 3100, Train accuracy: 90.0000%, Cross entropy: 0.723296, Validation accuracy: 89.0% (N=100)
Step: 3200, Train accuracy: 93.0000%, Cross entropy: 0.571625, Validation accuracy: 88.0% (N=100)
Step: 3300, Train accuracy: 89.0000%, Cross entropy: 0.759010, Validation accuracy: 82.0% (N=100)
Step: 3400, Train accuracy: 89.0000%, Cross entropy: 0.688089, Validation accuracy: 86.0% (N=100)
Step: 3500, Train accuracy: 88.0000%, Cross entropy: 0.692692, Validation accuracy: 90.0% (N=100)
Step: 3600, Train accuracy: 92.0000%, Cross entropy: 0.704201, Validation accuracy: 90.0% (N=100)
Step: 3700, Train accuracy: 93.0000%, Cross entropy: 0.681209, Validation accuracy: 87.0% (N=100)
Step: 3800, Train accuracy: 88.0000%, Cross entropy: 0.755828, Validation accuracy: 87.0% (N=100)
Step: 3900, Train accuracy: 87.0000%, Cross entropy: 0.676947, Validation accuracy: 83.0% (N=100)
Step: 4000, Train accuracy: 93.0000%, Cross entropy: 0.620935, Validation accuracy: 87.0% (N=100)
Step: 4100, Train accuracy: 93.0000%, Cross entropy: 0.665068, Validation accuracy: 91.0% (N=100)
Step: 4200, Train accuracy: 88.0000%, Cross entropy: 0.674545, Validation accuracy: 87.0% (N=100)
Step: 4300, Train accuracy: 85.0000%, Cross entropy: 0.700661, Validation accuracy: 89.0% (N=100)
Step: 4400, Train accuracy: 88.0000%, Cross entropy: 0.597467, Validation accuracy: 88.0% (N=100)
Step: 4500, Train accuracy: 91.0000%, Cross entropy: 0.593935, Validation accuracy: 97.0% (N=100)
Step: 4600, Train accuracy: 92.0000%, Cross entropy: 0.550575, Validation accuracy: 88.0% (N=100)
Step: 4700, Train accuracy: 91.0000%, Cross entropy: 0.566238, Validation accuracy: 92.0% (N=100)
Step: 4800, Train accuracy: 93.0000%, Cross entropy: 0.552427, Validation accuracy: 91.0% (N=100)
Step: 4900, Train accuracy: 92.0000%, Cross entropy: 0.600873, Validation accuracy: 88.0% (N=100)
Step: 4999, Train accuracy: 87.0000%, Cross entropy: 0.657480, Validation accuracy: 91.0% (N=100)
Final test accuracy = 90.3% (N=17852)
```

### Running the Classify with image

    python classify.py [options]
    
    Options:
        -i, --image-location: (Location of image to be classify.)
        
    Default Options: (If you dont write any options)
        -i dataset/A/A1.jpg

Expected Results:
```
a (score = 0.73810)
e (score = 0.16488)
i (score = 0.04245)
x (score = 0.01767)
z (score = 0.00997)
b (score = 0.00918)
m (score = 0.00597)
s (score = 0.00555)
f (score = 0.00184)
j (score = 0.00109)
t (score = 0.00101)
l (score = 0.00088)
k (score = 0.00036)
o (score = 0.00019)
n (score = 0.00018)
r (score = 0.00016)
y (score = 0.00012)
d (score = 0.00011)
u (score = 0.00009)
g (score = 0.00008)
space (score = 0.00005)
c (score = 0.00003)
w (score = 0.00002)
del (score = 0.00002)
v (score = 0.00001)
q (score = 0.00000)
h (score = 0.00000)
p (score = 0.00000)
nothing (score = 0.00000)

```

### Running the Classify via Webcam

    python classify_webcam.py
    
    
### Important Note

The dataset we use consists of approximately 87000 images. We were rejected while uploading our dataset on GitHub. That's why we share our drive link here.

https://drive.google.com/file/d/1B5xBQXDqG0EOt4vxiclpcV9eY21gp_jp/view?usp=sharing

You can put the dataset you downloaded from the drive into the root directory of your project.
