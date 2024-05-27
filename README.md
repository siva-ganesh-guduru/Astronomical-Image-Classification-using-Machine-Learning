(https://github.com/siva-ganesh-guduru/Astronomical-Image-Classification-using-Machine-Learning/assets/85402457/db4f51e2-2988-4ffe-9002-1a6d5d0e3052)# Astronomical-Image-Classification-using-Machine-Learning
This  is my Final year college B.tech Project
Contents • Abstract • Introduction • Problem Statement • Motivation • Objectives • Literature Survey • Limitations of existing methods • Physical model • Proposed algorithms • Software and Hardware requirements • Simulation model • Performance Evaluation • Analysis of Experimental Data

Abstract

Universe is a vast area of research that requires intensive investigation. Among millions of galaxies, finding out
whether the object is a planet, meteoroid, or some unusual stellar object gives greater insight into space research.
Astronomical image classification with machine learning means using computer programs to look at lots of
space pictures and figure out what's in them, like galaxies, stars, or planets. The programs learn to recognize
patterns and details in the images that might be hard for people to see. This helps astronomers study and make
sense of the huge amounts of data from telescopes and space probes, leading to discoveries in space. The data
we are using is taken from a survey for NEOs by Lori Allen and collaborators using DECam on the Blanco 4m
Telescope at CTIO. The data comprise a stack of images taken over a period of 5 nights.

Introduction

• Astronomical Image Classification with Machine Learning is like a mysterious telescope for researchers.
Particularly our work includes classification of point source images.
• First, we collect images categorized into two sets point source images and noise. Images are normalized to
circumvent the size limitations.
• Then, we instruct our computer programs, called Machine Learning models, to recognize what's within the
pictures, whether it’s a point source image or just some random noise.
• Rather than investing ages looking at each picture, they can consider space indeed more. But we have to keep
instructing our enchantment programs and work along with space specialists to form beyond any doubt we get
it right.

Problem statement

Dataset from a survey for NEOs by Lori Allen and collaborators using DECam on the Blanco 4m Telescope at
CTIO is categorized into point source images and noise. The task is to identify the best classification algorithm
and to make the predictions on the validation data set. The term point source generally means an observed
astronomical object that focuses as a point, it is a single identifiable localized source in the space.

Motivation

The motivations for our project align with the broader goals of applying machine learning to astronomical image
classification:
• Automating tedious tasks: Manually classifying numerous astronomical images is time-consuming and errorprone. Our project can contribute to automating this process, freeing up (astronomers, researchers, and anyone
involved) for more in-depth research and analysis.
• Uncovering hidden insights: By identifying subtle patterns or rare celestial objects, our project can potentially lead
to discoveries and improve our understanding of the universe. Imagine the groundbreaking research that could be
unlocked by unlocking the secrets hidden within these cosmic snapshots!
• Boosting accuracy and efficiency: Machine learning models can potentially achieve higher accuracy and
classify images much faster than humans, especially when dealing with large datasets

Objectives

1. To develop a robust system for classifying astronomical objects within large images.
2. To achieve a high precision rate and enhanced F1 score in the classification of astronomical objects, ensuring
accurate identification and categorization.
3. To design the system to be scalable, and capable of handling the vast amounts of data typically associated
with images in astronomy.
4. To evaluate and compare the performance of different classification algorithms such as GaussianNB,
Linear Discriminant Analysis, Quadratic Discriminant Analysis, Logistic Regression, KNeighbors
classifier, Decision tree classifier, and GMMBayes on astronomical images.
5. To address the issue of imbalanced datasets, which is common in astronomical data.

Literature Survey[cont.,]

1) Automatic classification of galaxy morphology: A rotationally-invariant supervised machine-learning
method based on the unsupervised machine-learning data set. GW Fang, S Ba, Y Gu, Z Lin, Y Hou, C
Qin, C Zhou, J Xu, Y Dai, J Song, X Kong. The Astronomical Journal, 2023•iopscience.iop.org
• .The proposed method employs adaptive polar-coordinate transformation, ensuring rotationally invariant
supervised machine learning (SML).
• This technique enhances classification consistency when rotating galaxy images, addressing a physical
requirement that is challenging algorithmically.
• Compared to traditional data augmentation methods, the adaptive polar-coordinate transformation proves to
be more effective and efficient in improving the robustness of SML methods.
Limitations: There is a lack of discussion on potential biases in the dataset or any efforts made to mitigate them,
raising concerns about the method's applicability to diverse astronomical scenarios.

Literature Survey

2) An Efficient Object Detection Method For Large CCD Astronomical Images. Jingchang Pan;Caiming
Zhang 2009 2nd International Congress on Image and Signal ProcessingYear: 2009 | Conference Paper |
Publisher: IEEE.
• The paper addresses the challenge of real-time processing of large CCD images in astronomy, stored in FITS
format, through the efficient detection and measurement of celestial objects.
• An innovative method is proposed, involving a scan accelerator and a recursive measure routine,
demonstrating fast and accurate results in object detection and measurement.
• The approach is easy to implement and has potential applicability in other related fields of study.
Limitations: The paper mainly focuses on star detection in CCD images. Limitations or considerations for
identifying other celestial bodies, like galaxies or asteroids, are not explicitly addressed.

Literature Survey[cont.,]

3) Resolving the celestial classification using fine k-NN classifier. Sangeeta Yadav;Amandeep Kaur;Neeraj
Singh Bhauryal. 2016 Fourth International Conference on Parallel, Distributed and Grid Computing
(PDGC)Year: 2016 | Conference Paper | Publisher: IEEE
• The abstract highlights the increasing volume of data from space missions and the need for automated
techniques to classify celestial objects in images.
• The paper proposes an artificial neural network-based classifier for the image classification of celestial
bodies, specifically planets.
• Texture features are extracted from a dataset of 90 images, and different classifiers, including k-NN and Fine
KNN, are applied to determine the most effective for space data classification.
Limitations: While the paper focuses on classifying planets, it may not address the broader range of celestial
objects, limiting the applicability of the proposed technique to other types of space data.

Literature Survey[cont.,]

4) Neural nets and star/galaxy separation in wide field astronomical images. S. Andreon;G. Gargiulo;G.
Longo;R. Tagliaferri;N. Capuano. IJCNN'99. International Joint Conference on Neural Networks.
Proceedings (Cat. No.99CH36339)Year: 1999 | Conference Paper | Publisher: IEEE
• The paper addresses challenges in extracting scientifically useful information from wide-field astronomical
images, particularly the recognition and classification of objects against a noisy background.
• A neural network-based method is introduced to perform both object detection and star/galaxy separation.
• The focus is on experimental results related to object detection, comparing the performance of this method
with other commonly used methodologies in the astronomical community.
Limitations: It focuses mainly on experimental results related to object detection, with less emphasis on
star/galaxy separation. This might limit the comprehensive evaluation of the neural network's capabilities..

Literature Survey[cont.,]

5) Self-supervised Learning for Astronomical Image Classification. Ana Martinazzo;Mateus
Espadoto;Nina S. T. Hirata. 2020 25th International Conference on Pattern Recognition
(ICPR)Year: 2021 | Conference Paper | Publisher: IEEE
• This paper addresses the challenge of limited labeled data in astronomy by proposing a technique that
leverages unlabeled astronomical images for pre-training deep convolutional neural networks (CNNs).
• In this approach, a large neural network is pre-trained with unlabeled data, where astronomical
properties are computed as regression targets based on the input images.
• This pre-training aims to create a domain-specific feature extractor.
Limitations: The paper mentions outperforming ImageNet pre-training in most cases but does not elaborate
on specific metrics used for comparison, making it important to understand the nuances of performance
evaluation.

Literature Survey[cont.,]

6) CzSL: Learning from citizen science, experts, and unlabelled data in astronomical image classification.
M Jiménez Morales, EJ Alfaro, M Torres Torres - 2023 - digibug.ugr.es
• This paper addresses the challenges and opportunities presented by citizen science in the context of
astronomical image classification, specifically focusing on the Galaxy Zoo project.
• Citizen science involves amateur participants contributing to scientific research by classifying large
collections of images.
• The paper proposes an innovative learning methodology that combines expert- and amateur-labeled data,
leveraging unlabelled data within a unified framework.
Limitations: The paper acknowledges the potential variability in skills and motivations among amateur
participants, but the extent to which this impacts classification accuracy and reliability is not extensively
explored.

Limitations of exiting methods

Here are some key limitations of existing methods that our project could address:
• Rotation invariance: One of the existing methods lacks rotation invariance, leading to inconsistencies in
classifying rotated galaxy images.
• Limited scope: Some researchers focus only on specific object types, like stars or planets, neglecting the
broader range of celestial bodies.
• Limited focus on specific tasks: Some papers prioritize object detection over classification or vice versa.
• Evaluation nuances: Several papers mention performance comparisons but lack clear details on the specific
metrics used.

Proposed algorithms

• K-nearest Neighbours (KNN): It classifies the data point on how its neighbour is classified. KNN classifies
the new data points based on the similarity measure of the earlier stored points.
• Linear Discriminant Analysis (LDA): The primary goal of LDA is to find the linear combinations of
features that best separate different classes while minimizing the variance within each class.
• Quadratic Discriminant Analysis (QDA): The main goal of QDA is to find quadratic decision boundaries
that best separate different classes based on their feature vectors.
• Convolutional Neural Networks (CNN): CNN is a type of artificial neural network used primarily for
image recognition and processing, due to its ability to recognize patterns in images. In CNN every image is
represented in the form of an array of pixel values. The convolution operation form the basis of any
convolutional neural network.
• You Only Look Once (YOLO): Yolo proposes an end to end neural network that make predictions of
bounding boxes and class probabilities all at once. It detects multiple objects in single image by partitioning
the image into an S x S grid, with each grid cell responsible for detecting objects whose centers fall within it.

Software and hardware requirements:

Software requirements:

• Language:python 3.6 or higher specifiations
• Packages: Numpy, sklearn ,matplotlib,astroML
• Operating Systems: windows 10

Hardware requirements:

• Ram minimum 4gb
• Hard disc or ssd : abov e 500 gb


Performance Evaluation

• KNN classifier:

Accuracy: 0.8285779502396713
Classification Report:
precision recall f1-score support
0 0.81 0.90 0.85 11996
1 0.86 0.75 0.80 9909

Confusion Matrix:

[[10747 1249]
[ 2506 7403]]

• Linear Discriminant Analysis (LDA) :

Accuracy: 0.8931294225062771

Classification Report:

precision recall f1-score support
0 0.91 0.89 0.90 11996
1 0.87 0.89 0.88 9909

Confusion Matrix:

[[10715 1281]
[ 1060 8849]]

Performance Evaluation

• Quadratic Discriminant Analysis (QDA) :

Accuracy: 0.6806208628167085

Classification Report:

precision recall f1-score support
0 0.63 1.00 0.77 11996
1 1.00 0.29 0.45 9909

Confusion Matrix:

[[11995 1]
[ 6995 2914]]

• Convolutional neural network (CNN) :

Accuracy: 0.9775
Classification Report:
precision recall f1-score support
0.0 0.97 0.99 0.98 12000
1.0 0.99 0.96 0.97 9905

Confusion Matrix:

[[11893 107]
[ 386 9519]]

Performance Evaluation

• You Only Look Once (YOLO):

Accuracy: 0.9860762357711792
Classification Report:
precision recall f1-score support
0 0.99 0.99 0.99 12023
1 0.98 0.99 0.98 9882

Confusion Matrix:

[[11139 857]
[ 234 9675]]

Analysis of Experimental Data

The training sample includes images of simulated TNOs (true positives; stamps_sources.npz) and random
trajectories where there is no known source (false positives; stamps_noise.npz). The true positives range in
signal-to-noise from 100 to 3.


