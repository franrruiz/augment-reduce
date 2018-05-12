# Augment and Reduce

Code for Augment &amp; Reduce, a scalable stochastic algorithm for large categorical distributions.

This code replicates the experiments in the paper
+ Francisco J. R. Ruiz, Michalis K. Titsias, Adji B. Dieng, and David M. Blei. *Augment and Reduce: Stochastic Inference for Large Categorical Distributions*. International Conference on Machine Learning. 2018.

This code trains a linear multiclass classifier on a dataset with a large number of classes.

Please cite this paper if you use this software.


## Compilation

The code is written in Matlab, combined with C++ functions.

The flag `flag_mexFile` controls whether you wish to use the C++ code. It is strongly recommended to leave the flag active to speed up the code. This is the default setting. For that, you first need to compile the mex files using the two steps below (they should work under Mac and Unix).

+ First, make sure you have the [GSL library](https://www.gnu.org/software/gsl) installed. If so, open a terminal and run
```
gsl-config --cflags --libs
```
Copy the output on the clipboard; you will need it for the second step.

+ Second, open Matlab, `cd` to the repo path, and run the commands below, replacing `<TERMINAL_OUTPUT>` with the output from Step 1.
```
    mex CFLAGS="\$CFLAGS" -largeArrayDims src/infer/compute_psi.cpp -outdir src/infer
    mex CFLAGS="\$CFLAGS" -largeArrayDims src/infer/increase_follow_gradients.cpp -outdir src/infer
    mex CFLAGS="\$CFLAGS" -largeArrayDims src/aux/keep_first_label_c.cpp -outdir src/aux
    mex CFLAGS="\$CFLAGS" <TERMINAL_OUTPUT> -largeArrayDims src/infer/compute_predictions_c.cpp -outdir src/infer
    mex CFLAGS="\$CFLAGS" <TERMINAL_OUTPUT> src/aux/multirandperm.cpp -outdir src/aux
```

## Data Format

The data should be contained in a Matlab struct object and it must contain the following fields:

```
data          a struct containing the data
data.X        the training data (instances x dimensions). It MUST be in sparse matrix format (use the command sparse).
data.Y        the training labels (instances x 1). Each element indicates the class (from 1, ..., K).
data.test     a struct containing the test data
data.test.X   the test data (test_instances x dimensions). It MUST be in sparse matrix format.
data.test.Y   the test labels (test_instances x 1). Each element indicates the class (from 1, ..., K).
```

Please refer to the main files in `src/` for additional information.
