# rowing-practice-prediction

I have been a passionate rowing athlete since I picked it up in the 8th grade. This project serves as a perfect intersection between my passion for rowing and for solving problems with technical solutions.

I finished SENG 474: Data Mining in the Fall of 2025. In this course we used the SciKitLearn library to train classifiers, perform text classification, clustering, and more. This powerful tool was so exciting to learn.

It was the perfect intersection of these two aspects of my life when one of my crew members mentioned how cumbersome it is to have to wait until we are physically at the lake in order to make the call if practice will happen or not.

The lake I row on, Elk Lake, takes into account several weather factors: temperature, visibility, wind, and lightning. My goal was to train a classifer on Elk Lake data to be able to predict wether practice would happen or not, thus avoiding having to drive all the way out to the lake just for practice to be called off.

My Process

I began by collecting data based off of practices where I rowed or didn't row, while also recording the conditions.
I then implemented a standard classifier using SciKitLearn in a Jupyter notebook using most of this data for training, and the remaining data for testing.
I finally evaluated the models accuracy on the test set, before making future predictions.

Future Iterations

As rowing continues in the new year I will be collecting more data. My goal is to collect enough data to perform a cross validation using 10 folds, and thus improve its accuracy.

