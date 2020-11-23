# ImageMetrics
A repository used for various binary metrics used to evaluate (especially) medical images

Currently the repository contains.

* Confusion matrix
* F1 Score
* Object F1 Score  (The object-level Dice index)
* Hausdorff Distance¹
* Object Hausdorff Distance¹ (object-level Hausdorff distance)

The metrics are inspired by https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation

¹The Hausdroff distance is not defined for all zero masks. In case of all zero mask, we define this metric to yield be the longest distance found by the pythagoras theorem. This converstion is also not how they define the metric in the glascontest.
