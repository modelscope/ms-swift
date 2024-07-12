import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint, load_sharded_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
# from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata
# from megatron.training.utils import get_ltor_masks_and_position_ids
import sys
from argparse import Namespace

@dataclass
class MegatronArguments:
    # model
    num_layers: int = 24
    hidden_size: int = 896
    ffn_hidden_size: hidden_size = 4864  # intermediate_size
    num_attention_heads: int = 14
    group_query_attention: bool = True
    num_query_groups: int = 2
    max_position_embeddings: int = 131072
    position_embedding_type: str = 'rope'
    use_rotary_position_embeddings: bool = True
    rotary_percent: float = 1.
    rotary_seq_len_interpolation_factor: int = 1
    normalization: str = 'RMSNorm'
    norm_epsilon: float = 1e-6
    swiglu: bool = True
    untie_embeddings_and_output_weights: bool = True
    attention_dropout: float = 0.
    hidden_dropout: float = 0.
    # train
    weight_decay: float = 0.1
    clip_grad: float = 1.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    micro_batch_size: int = 1
    global_batch_size: int = 8
    recompute_method: Optional[str] = None
    recompute_granularity: Optional[str] = 'selective'
    train_iters: int = 1000
    train_samples: Optional[int] = 0
    log_interval: int = 10
    tensorboard_dir: str= 'output/tensorboard'
    save: str = 'output'
    apply_rope_fusion: bool = False
    use_flash_attn: bool = False
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    optimizer: str = 'adam'
    dataloader_type: str = 'cyclic'
    async_tensor_model_parallel_allreduce: bool = True
    sequence_parallel: bool = False
    seed: int = 1234
    init_method_std: float = 0.008
    lr: float = 1e-5
    lr_decay_style: str = 'cosine'

    target_tensor_model_parallel_size: int = 1
    target_pipeline_model_parallel_size: int = 1


    # 


    def get_megatron_args(self) -> Namespace:
        new_args = []

        sys._old_argv = sys.argv
        sys.argv = sys._old_argv[:1] + new_args

        initialize_megatron()
        return get_args()

if __name__ == '__main__':
    args = MegatronArguments()
    megatron_args = args.get_megatron_args()

    print()


"""
lr=1e-05, lr_decay_style='cosine', lr_decay_iters=200, lr_decay_samples=None, lr_warmup_fraction=None, lr_warmup_iters=100, lr_warmup_samples=0, lr_warmup_init=0.0, min_lr=1e-06, override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=False, decoupled_lr=None, decoupled_min_lr=None, save='abc', save_interval=1, no_save_optim=None, no_save_rng=None, load='qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1', no_load_optim=None, no_load_rng=None, finetune=False, pretrained_checkpoint=None, ckpt_step=None, perform_initialization=True, use_checkpoint_args=False, exit_on_missing_checkpoint=False, use_dist_ckpt=False, auto_detect_ckpt_format=False, dist_ckpt_format='torch_dist', ckpt_fully_parallel_save=False, async_save=None, ckpt_fully_parallel_load=False, ckpt_assume_constant_structure=False, fp16=False, bf16=False, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, fp32_residual_connection=False, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=False, accumulate_allreduce_grads_in_fp32=False, fp16_lm_cross_entropy=False, tensor_model_parallel_size=1, pipeline_model_parallel_size=1, pipeline_model_parallel_split_rank=None, num_layers_per_virtual_pipeline_stage=None, overlap_p2p_comm=False, distributed_backend='nccl', distributed_timeout_minutes=10, overlap_grad_reduce=False, delay_grad_reduce=True, ddp_bucket_size=None, ddp_average_in_collective=False, overlap_param_gather=False, delay_param_gather=False, scatter_gather_tensors_in_pipeline=True, use_ring_exchange_p2p=False, local_rank=0, lazy_mpu_init=None, standalone_embedding_stage=False, use_distributed_optimizer=False, context_parallel_size=1, nccl_communicator_config_path=None, use_tp_pp_dp_mapping=False, eval_iters=100, eval_interval=1000, test_mode=False, skip_train=False, data_path=None, split=None, train_data_path=['qwen-datasets/alpaca_zh-qwen-train.json'], valid_data_path=['qwen-datasets/alpaca_zh-qwen-valid.json'], test_data_path=['qwen-datasets/alpaca_zh-qwen-valid.json'], data_cache_path=None, mmap_bin_files=True, mock_data=False, vocab_size=-1, vocab_file=None, merge_file=None, vocab_extra_ids=0, seq_length=128, encoder_seq_length=128, decoder_seq_length=None, retriever_seq_length=256, sample_rate=1.0, mask_prob=0.15, short_seq_prob=0.1, num_workers=2, tokenizer_type='NullTokenizer', tokenizer_model=None, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=False, create_attention_mask_in_dataloader=True, num_dataset_builder_threads=1, adlr_autoresume=False, adlr_autoresume_interval=1000, ict_head_size=None, biencoder_projection_dim=0, biencoder_shared_query_context_model=False, ict_load=None, bert_load=None, titles_data_path=None, query_in_block_prob=0.1, use_one_sent_docs=False, evidence_data_path=None, retriever_report_topk_accuracies=[], retriever_score_scaling=False, block_data_path=None, embedding_path=None, indexer_batch_size=128, indexer_log_interval=1000, num_classes=1000, img_h=224, img_w=224, num_channels=3, patch_dim=16, classes_fraction=1.0, data_per_class_fraction=1.0, data_sharding=True, head_lr_mult=1.0, vision_pretraining=False, vision_pretraining_type='classify', vision_backbone_type='vit', swin_backbone_type='tiny', mask_type='random', mask_factor=1.0, iter_per_epoch=1250, dino_local_img_size=96, dino_local_crops_number=10, dino_head_hidden_size=2048, dino_bottleneck_size=256, dino_freeze_last_layer=1, dino_norm_last_layer=False, dino_warmup_teacher_temp=0.04, dino_teacher_temp=0.07, dino_warmup_teacher_temp_epochs=30, qk_layernorm=False, expert_model_parallel_size=1, num_experts=None, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_grouped_gemm=False, moe_aux_loss_coeff=0.0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dispatcher_type='allgather', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_extended_tp=False, log_params_norm=False, log_num_zeros_in_grad=False, log_throughput=False, log_progress=False, timing_log_level=0, barrier_with_L1_time=True, timing_log_option='minmax', tensorboard_log_interval=1, tensorboard_queue_size=1000, log_timers_to_tensorboard=False, log_batch_size_to_tensorboard=False, log_learning_rate_to_tensorboard=True, log_loss_scale_to_tensorboard=True, log_validation_ppl_to_tensorboard=False, log_memory_to_tensorboard=False, log_world_size_to_tensorboard=False, wandb_project='', wandb_exp_name='', wandb_save_dir='', enable_one_logger=False, one_logger_project='e2e-tracking', one_logger_entity='hwinf_dcm', one_logger_run_name=None, logging_level=None, log_straggler=False, disable_straggler_on_startup=False, straggler_ctrlr_port=65535, straggler_minmax_count=1, inference_batch_times_seqlen_threshold=512, max_tokens_to_oom=12000, output_bert_embeddings=False, bert_embedder_type='megatron', fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, transformer_impl='transformer_engine', retro_project_dir=None, retro_add_retriever=False, retro_cyclic_train_iters=None, retro_encoder_layers=2, retro_encoder_hidden_dropout=0.1, retro_encoder_attention_dropout=0.1, retro_num_neighbors=2, retro_num_retrieved_chunks=2, retro_attention_gate=1, retro_verify_neighbor_count=True, spec=None, yaml_cfg=None, n_head_kv=None, transformer_type='megatron', max_padding_length=128, dataset='LLama-Pretrain-Raw', epochs=None, intermediate_size=None, extra_vocab_size=293, keep_last=False, data_dir=None, train_data=None, valid_data=None, patch_tokenizer_type='Qwen2Tokenizer', use_alibi_mask=False, use_normhead=False, glu_activation=None, attention_head_type=None, transformer_timers=False, text_generate_input_file='', text_generate_output_file='', text_generate_gt_file='', time=False, eval_dev=False, input_len=1, generation_length=None, top_p=0.0, top_k=0, out_seq_length=1024, temperature=1.0, repetition_penalty=1.1, embed_layernorm=False, source_seq_len=None, target_seq_len=None, position_encoding_2d=False, z_loss_weight=0.0, use_llama2_rotary_position_embeddings=False, use_mistral_rotary_position_embeddings=False, mm_use_im_start_end=False, mm_use_im_patch_token=False, tune_mm_mlp_adapter=False, freeze_clip_vision_tower=False, freeze_llm=False, image_folder='', mm_vision_select_layer=None, vision_tower='', image_aspect_ratio='square', version='plain', mm_projector_type=None, image_size=None, patch_size=None, sliding_window=None, rotary_base=10000, rotary_scale_factor=1, cvcuda_image_processing=False, expert_tensor_parallelism=False, expert_interval=2, moe=False, moe_topk=1, moe_expert_parallel_size=None, moe_train_capacity_factor=1.0, moe_eval_capacity_factor=1.0, moe_min_capacity=4, moe_loss_coeff=0.01, use_tutel=False, router_type='topk', moe_input_feature_slicing=False, add_bias_linear_fc=True, add_bias_attn_fc=True, enable_parallel_output=True, task_list='all', verbosity='INFO', adaptive_seq_len=False, eval_fp32=False, num_fewshot=None, convert_checkpoint_from_megatron_to_transformers=False, moe_ffn_hidden_size=None, shared_moe_ffn_hidden_size=None, enable_shared_expert=False, q_lora_rank=None, kv_lora_rank=None, qk_nope_head_dim=None, qk_rope_head_dim=None, v_head_dim=None, num_shared_experts=None, moe_layer_freq=1, rotary_scaling_factor=1, rank=0, world_size=2, transformer_pipeline_model_parallel_size=1, data_parallel_size=2, virtual_pipeline_model_parallel_size=None, params_dtype=torch.float32, consumed_train_samples=0, consumed_valid_samples=0, variable_seq_lengths=False, padded_vocab_size=0, model_type=<ModelType.encoder_or_decoder: 1>)

lr=1e-05, lr_decay_style='cosine', lr_decay_iters=900, lr_decay_samples=None, lr_warmup_fraction=None, lr_warmup_iters=100, lr_warmup_samples=0, lr_warmup_init=0.0, min_lr=1e-06, override_opt_param_scheduler=False, use_checkpoint_opt_param_scheduler=False, decoupled_lr=None, decoupled_min_lr=None, save='../../output_mcore_qwen/checkpoint/dsw-finetune-mcore-qwen2-0.5B-lr-1e-5-bs-1-seqlen-128-pr-bf16-tp-1-pp-1-ac-sel-do-true-sp-false-tt--wt-', save_interval=100, no_save_optim=None, no_save_rng=None, load='../../qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1', no_load_optim=True, no_load_rng=True, finetune=False, pretrained_checkpoint=None, ckpt_step=None, perform_initialization=True, use_checkpoint_args=False, exit_on_missing_checkpoint=False, use_dist_ckpt=False, auto_detect_ckpt_format=False, dist_ckpt_format='torch_dist', ckpt_fully_parallel_save=False, async_save=None, ckpt_fully_parallel_load=False, ckpt_assume_constant_structure=False, fp16=False, bf16=True, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, fp32_residual_connection=False, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=False, accumulate_allreduce_grads_in_fp32=True, fp16_lm_cross_entropy=False, tensor_model_parallel_size=1, pipeline_model_parallel_size=1, pipeline_model_parallel_split_rank=None, num_layers_per_virtual_pipeline_stage=None, overlap_p2p_comm=False, distributed_backend='nccl', distributed_timeout_minutes=10, overlap_grad_reduce=False, delay_grad_reduce=True, ddp_bucket_size=None, ddp_average_in_collective=False, overlap_param_gather=False, delay_param_gather=False, scatter_gather_tensors_in_pipeline=True, use_ring_exchange_p2p=False, local_rank=1, lazy_mpu_init=None, standalone_embedding_stage=False, use_distributed_optimizer=True, context_parallel_size=1, nccl_communicator_config_path=None, use_tp_pp_dp_mapping=False, eval_iters=10, eval_interval=10000, test_mode=False, skip_train=False, data_path=None, split=None, train_data_path=['../../qwen-datasets/alpaca_zh-qwen-train.json'], valid_data_path=['../../qwen-datasets/alpaca_zh-qwen-valid.json'], test_data_path=['../../qwen-datasets/alpaca_zh-qwen-valid.json'], data_cache_path=None, mmap_bin_files=True, mock_data=False, vocab_size=-1, vocab_file=None, merge_file=None, vocab_extra_ids=0, seq_length=128, encoder_seq_length=128, decoder_seq_length=None, retriever_seq_length=256, sample_rate=1.0, mask_prob=0.15, short_seq_prob=0.1, num_workers=8, tokenizer_type='NullTokenizer', tokenizer_model=None, reset_position_ids=False, reset_attention_mask=False, eod_mask_loss=True, create_attention_mask_in_dataloader=True, num_dataset_builder_threads=1, adlr_autoresume=False, adlr_autoresume_interval=1000, ict_head_size=None, biencoder_projection_dim=0, biencoder_shared_query_context_model=False, ict_load=None, bert_load=None, titles_data_path=None, query_in_block_prob=0.1, use_one_sent_docs=False, evidence_data_path=None, retriever_report_topk_accuracies=[], retriever_score_scaling=False, block_data_path=None, embedding_path=None, indexer_batch_size=128, indexer_log_interval=1000, num_classes=1000, img_h=224, img_w=224, num_channels=3, patch_dim=16, classes_fraction=1.0, data_per_class_fraction=1.0, data_sharding=True, head_lr_mult=1.0, vision_pretraining=False, vision_pretraining_type='classify', vision_backbone_type='vit', swin_backbone_type='tiny', mask_type='random', mask_factor=1.0, iter_per_epoch=1250, dino_local_img_size=96, dino_local_crops_number=10, dino_head_hidden_size=2048, dino_bottleneck_size=256, dino_freeze_last_layer=1, dino_norm_last_layer=False, dino_warmup_teacher_temp=0.04, dino_teacher_temp=0.07, dino_warmup_teacher_temp_epochs=30, qk_layernorm=False, expert_model_parallel_size=1, num_experts=None, moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_grouped_gemm=False, moe_aux_loss_coeff=0.0, moe_z_loss_coeff=None, moe_input_jitter_eps=None, moe_token_dispatcher_type='allgather', moe_per_layer_logging=False, moe_expert_capacity_factor=None, moe_pad_expert_input_to_capacity=False, moe_token_drop_policy='probs', moe_layer_recompute=False, moe_extended_tp=False, log_params_norm=False, log_num_zeros_in_grad=False, log_throughput=False, log_progress=False, timing_log_level=0, barrier_with_L1_time=True, timing_log_option='minmax', tensorboard_log_interval=1, tensorboard_queue_size=1, log_timers_to_tensorboard=True, log_batch_size_to_tensorboard=True, log_learning_rate_to_tensorboard=True, log_loss_scale_to_tensorboard=True, log_validation_ppl_to_tensorboard=True, log_memory_to_tensorboard=False, log_world_size_to_tensorboard=False, wandb_project='', wandb_exp_name='', wandb_save_dir='', enable_one_logger=False, one_logger_project='e2e-tracking', one_logger_entity='hwinf_dcm', one_logger_run_name=None, logging_level=None, log_straggler=False, disable_straggler_on_startup=False, straggler_ctrlr_port=65535, straggler_minmax_count=1, inference_batch_times_seqlen_threshold=512, max_tokens_to_oom=12000, output_bert_embeddings=False, bert_embedder_type='megatron', fp8=None, fp8_margin=0, fp8_interval=1, fp8_amax_history_len=1, fp8_amax_compute_algo='most_recent', fp8_wgrad=True, transformer_impl='transformer_engine', retro_project_dir=None, retro_add_retriever=False, retro_cyclic_train_iters=None, retro_encoder_layers=2, retro_encoder_hidden_dropout=0.1, retro_encoder_attention_dropout=0.1, retro_num_neighbors=2, retro_num_retrieved_chunks=2, retro_attention_gate=1, retro_verify_neighbor_count=True, spec=None, yaml_cfg=None, n_head_kv=None, transformer_type='megatron', max_padding_length=128, dataset='LLama-Pretrain-Raw', epochs=None, intermediate_size=None, extra_vocab_size=293, keep_last=False, data_dir=None, train_data=None, valid_data=None, patch_tokenizer_type='Qwen2Tokenizer', use_alibi_mask=False, use_normhead=False, glu_activation=None, attention_head_type=None, transformer_timers=False, text_generate_input_file='', text_generate_output_file='', text_generate_gt_file='', time=False, eval_dev=False, input_len=1, generation_length=None, top_p=0.0, top_k=0, out_seq_length=1024, temperature=1.0, repetition_penalty=1.1, embed_layernorm=False, source_seq_len=None, target_seq_len=None, position_encoding_2d=False, z_loss_weight=0.0, use_llama2_rotary_position_embeddings=False, use_mistral_rotary_position_embeddings=False, mm_use_im_start_end=False, mm_use_im_patch_token=False, tune_mm_mlp_adapter=False, freeze_clip_vision_tower=False, freeze_llm=False, image_folder='', mm_vision_select_layer=None, vision_tower='', image_aspect_ratio='square', version='plain', mm_projector_type=None, image_size=None, patch_size=None, sliding_window=None, rotary_base=1000000, rotary_scale_factor=1, cvcuda_image_processing=False, expert_tensor_parallelism=False, expert_interval=2, moe=False, moe_topk=1, moe_expert_parallel_size=None, moe_train_capacity_factor=1.0, moe_eval_capacity_factor=1.0, moe_min_capacity=4, moe_loss_coeff=0.01, use_tutel=False, router_type='topk', moe_input_feature_slicing=False, add_bias_linear_fc=True, add_bias_attn_fc=True, enable_parallel_output=True, task_list='all', verbosity='INFO', adaptive_seq_len=False, eval_fp32=False, num_fewshot=None, convert_checkpoint_from_megatron_to_transformers=False, moe_ffn_hidden_size=None, shared_moe_ffn_hidden_size=None, enable_shared_expert=False, q_lora_rank=None, kv_lora_rank=None, qk_nope_head_dim=None, qk_rope_head_dim=None, v_head_dim=None, num_shared_experts=None, moe_layer_freq=1, rotary_scaling_factor=1, rank=1, world_size=4, transformer_pipeline_model_parallel_size=1, data_parallel_size=4, virtual_pipeline_model_parallel_size=None, params_dtype=torch.bfloat16, consumed_train_samples=0, consumed_valid_samples=0, variable_seq_lengths=False, padded_vocab_size=0, model_type=<ModelType.encoder_or_decoder: 1>)
"""

"""

"""