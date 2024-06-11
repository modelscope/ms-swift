# Copyright (c) Alibaba, Inc. and its affiliates.
import custom

from swift.llm import rlhf_main
"""
This script runs the main process of Human Preferences Alignment using different algorithms.
Specify the algorithm using the --rlhf_type argument.

Options:
    --rlhf_type dpo    : Direct Preference Optimization (DPO)
    --rlhf_type kto    : Kahneman-Tversky Optimization (KTO)
    --rlhf_type cpo    : Contrastive Preference Optimization (CPO)
    --rlhf_type simpo  : Simple Preference Optimization (SimPO)
    --rlhf_type orpo   : Odds Ratio Preference Optimization (ORPO)
"""

if __name__ == '__main__':
    output = rlhf_main()
