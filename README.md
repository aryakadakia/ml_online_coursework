# Machine learning practice

Self-directed practice working through model-building APIs in TensorFlow and
Keras, and a set of scikit-learn workflows on small tabular datasets.

## TensorFlow and Keras

The first four notebooks build the same class of model three different ways, to
understand what each API gives up and what it buys.

| File | Focus |
|---|---|
| `SequentialModel.ipynb` | Sequential API on a heart disease dataset, from preprocessing to evaluation |
| `FunctionalModel.ipynb` | Functional API, allowing branching and multiple inputs |
| `ModelSubclassing.ipynb` | Subclassing `Model` for full control of the forward pass |
| `GradientTape.ipynb` | Automatic differentiation, and writing a training loop by hand |
| `TensorsAndVariables.py` | Tensor operations and variable semantics |

## scikit-learn

| File | Method |
|---|---|
| `HyperparameterTuningWithGridSearch.py` | Grid search with precision and recall scoring |
| `MultipleRegressionModels_Automobile.py` | Multiple regression with `statsmodels` diagnostics |
| `MLPClassification_LowerBackPain.py` | Multilayer perceptron classification |
| `KMeansAndMiniBatchClustering_Images.py` | K-means and mini-batch k-means for image colour quantisation |
| `RBMUnsupervisedDimensionalityReduction_MNISTClassification.py` | Restricted Boltzmann machine features feeding logistic regression on MNIST |
| `SimpleLinearRegression.ipynb` | Linear regression from first principles |

## Data

Datasets are read from a sibling `datasets/` directory and are not included.

## Stack

`tensorflow`, `keras`, `scikit-learn`, `statsmodels`, `pandas`.
