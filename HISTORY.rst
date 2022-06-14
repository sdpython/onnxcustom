
.. _l-HISTORY:

=======
History
=======

current - 2022-05-30 - 0.00Mb
=============================

* #64: Fixes bug introduced by recent updates (2022-05-29)
* #62: Adds a notebook about convolution (2022-05-27)
* #63: Add GraphProto to the documentation (2022-04-28)
* #60: Extends notebook coverage (2022-03-07)
* #59: Removes unnecessary exceptions (2022-03-06)
* #58: Uses new API to retrieve gradient for a model (2022-03-03)
* #57: Fix opset for ai.onnx.ml in examples after updating to onnx==1.11 (2022-02-22)
* #56: Renames Z into Y_grad in loss functions returning grad (2022-02-14)
* #55: Removes unncessary nodes in onnx_derivative (2022-02-14)
* #54: Implements a function which returns the gradient (2022-02-13)

0.4.293 - 2022-02-11 - 0.07Mb
=============================

* #53: Implements get_trained_onnx to retrieve the trained model (2022-02-04)
* #51: Implements scoring functions (2022-02-02)
* #49: Check nan values during training (2022-01-30)

0.4.274 - 2022-01-29 - 0.06Mb
=============================

* #48: Checks gradients are identical to scikit-learn for neural networks (2022-01-25)
* #47: Compare gradient values with scikit-learn (2022-01-25)
* #45: Adds an example about classification (2022-01-23)
* #44: Uses bind_ortvalue_input instead of bind_input to be faster (2022-01-23)
* #32: Implements classificiation loss (2022-01-23)
* #43: Fixes the training of a binary classifier with weights (2022-01-22)
* #42: Implements binary log loss for first API of orttraining (2022-01-22)
* #41: Refactors to improve profiling analysis (fix penalty, export onnx graphs) (2022-01-15)
* #40: Improves training fwbw with caching. (2022-01-14)
* #37: Improves performance of caching (2022-01-09)
* #36: Makes sure all plt.show() have been disabled in examples (2022-01-04)
* #35: Reduces the number of calls to bind_ortvalue (2022-01-04)

0.3.245 - 2022-01-03 - 0.06Mb
=============================

* #34: Implements L1, L2 losses for partial training (2022-01-03)
* #31: Implements penalty when running the gradient (2022-01-03)
* #33: Implements more loss functions (2022-01-02)
* #30: Implements different learning rate strategies (2022-01-01)
* #26: Implements learning rate from neural network (2022-01-01)
* #29: Fixes #16, support weights while training (2021-12-30)
* #16: Support weights when training a model (2021-12-30)
* #28: Replaces OrtValue by C_OrtValue everywhere (2021-12-19)
* #24: Be more consistent with OrtValue, OrtDevice, C and python versions (2021-12-19)
* #27: Uses C_OrtDevice everywhere (2021-12-16)
* #15: Add example with TrainingAgent (error gradient outside) (2021-12-14)
* #25: Implements optimizers with forward, backward functionalities (2021-12-04)
* #21: Implements a mechanism that update training weights with SGDRegressor or MLPRegressor (2021-12-04)
* #23: Extend documentation (2021-12-01)
* #22: Move learning_rate logic in a separate class (2021-12-01)
* #20: Experiment markdown rendering (2021-12-01)
* #19: Implements training with forward, backward (2021-12-01)
* #18: Adds classes to train an ONNX gradient with TrainingAgent (2021-11-27)
* #17: Minimize the number of data copy while training a model (2021-11-26)
* #14: Optimize DataLoader to use iobinding (avoir copy) (2021-11-26)
* #13: Adds function plot_onnxs (2021-11-25)
* #12: Add more examples (2021-11-19)

0.2.122 - 2021-10-31 - 0.03Mb
=============================

0.2.117 - 2021-10-26 - 0.03Mb
=============================

* #11: Automates nvprof logs retrieval (2021-10-26)
* #10: Add parameter to evaluate the model on test data while training (2021-10-12)
* #9: Refactoring documentation (2021-10-08)
* #8: Add example to look into neural network on GPU (2021-10-07)
* #7: Refactoring (2021-10-04)
* #6: Add examples with orttraining (2021-10-04)
* #5: Fix examples, update documentation (2021-09-28)
* #4: Complex scenarios (2021-07-12)
* #3: Replaces OnnxSubOperator by OnnxSubEstimator (2021-03-31)

0.1.0 - 2020-07-09 - 0.04Mb
===========================

* #1: Add an example on black list, while list of operators. (2020-07-09)
