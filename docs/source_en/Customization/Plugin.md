# Pluginization

Pluginization is an important capability introduced in SWIFT 3.0. We hope to allow developers to customize the development process more naturally through pluginization.

## callback

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/callback.py).

Callbacks are registered into the trainer before constructing the trainer. The example provides a simple version of the EarlyStop scheme.

## Customized Trainer

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/custom_trainer.py).

Users can inherit existing trainers and implement their own training logic here, such as customizing data loaders, customizing compute_loss, etc. The example demonstrates a trainer for a text-classification task.

## Customized Loss

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/loss.py).

Users can customize their own loss implementation plan here.

## Customized Loss Scale

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/loss_scale.py).

Users can customize the loss scale here, allowing for different weight definitions for different tokens.

## Customized Metric

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/metric.py).

Users can customize evaluation metrics used during the cross-validation phase here.

## Customized Optimizer

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/optimizer.py).

Users can add their own optimizer and lr_scheduler implementations here.

## Customized Tools

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/tools.py).

Users can add tools formatted for Agents here.

## Customized Tuner

Examples can be found [here](https://github.com/modelscope/swift/blob/main/swift/plugin/tuner.py).

Users can add new custom tuners here.
