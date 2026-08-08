# Copyright (c) Alibaba, Inc. and its affiliates.

from swift.loss import loss_map

from loss_dev import RouteHybridInfonceLoss

loss_map['route_hybrid_infonce'] = RouteHybridInfonceLoss
