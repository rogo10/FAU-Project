# FAU-Project

The notebook with the models predicts images that contain ditylum and those with no ditylum in them. This was the starting point for model building.

The rotate images file was used to create more non-ditylum photos to bring the count to 1330 non-ditylum photos to 1729 ditylum photos.

resize.py takes all photos and makes them the same shape (500x500).

cnn_copy.py was an attempt at a convolutional neural network but accuracy was only ~60%.

fau_clean_copy.py currently has 3 models with at least 88% accuracy.
