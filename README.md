# ImageMetrics
A repository used for various binary metrics used to evaluate (especially) medical images.

All the metrics are implementet with numpy and scikit-image. 

Currently the repository contains:

* Object count ratio
* Confusion matrix
* F1 Score
* IoU Score
* MCC Score
* Hausdorff Distance¹

The scores F1, IoU and Hausdorff have a object based implementation available.
 
The metrics are inspired by https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation

¹The Hausdroff distance is not defined for all zero masks. In case of all zero mask, we define this metric to yield be the longest distance found by the pythagoras theorem. This converstion is also not how they define the metric in the glascontest.
