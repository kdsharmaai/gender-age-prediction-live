# POC AI Projects

## Human gender and age prediction (CNN, Computer Vision)

Dataset used: [UTKFace kaggle dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new) 

Total more than 23K images, From which below information extrated from image labels,
- Image name
- Gender (0-male/1-female)
- Age

Model processed with grayscale images for fast process time, and small image input shape used (128, 128, 1)

Model design with increase number of nodes, Max pooling layer, Dense and Dropout layers

Cross verification of predicted images by human verification

Human gender identity accuracy: 90% and age MAE: 6.5

Predicted with my own image and gave perfect accuracy with age absolute error: 1

Model saved as .h5 file and processed data (images numpy array, gender, age) saved as .npy file, and use it to reduce process time 

