def test_to_megatron():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    export_main(ExportArguments(model_type='qwen2-0_5b', to_megatron=True, check_model_forward=True))


def test_to_hf():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from swift.llm import export_main, ExportArguments
    export_main(
        ExportArguments(
            model_type='qwen2-0_5b', to_hf=True, ckpt_dir='/mnt/nas2/huangjintao.hjt/work/swift/qwen2-0_5b-tp1-pp1'))


def test_pretrain():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')
    from swift.llm import get_model_tokenizer
    from swift.llm.megatron import (load_megatron_config, MegatronArguments, convert_megatron_to_hf,
                                    get_model_seires, patch_megatron, model_provider)
    res = {
        'load': '/mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1-new',
        'bf16': True,
    }

    model_type = 'qwen2-0_5b'
    _, tokenizer = get_model_tokenizer(model_type, load_model=False)
    res = load_megatron_config(tokenizer.model_dir)
    res['model_series'] = get_model_seires(model_type)
    res.update({
        'train_iters': 1000,
        'eval_iters': 100,
        'lr_warmup_iters': 100,
        'save': 'output/megatron',
        'tensorboard_dir': 'output/megatron/runs',
        'bf16': True,
        'load': '/mnt/nas2/huangjintao.hjt/work/swift/qwen2-0_5b-tp1-pp1',
    })
    megatron_args = MegatronArguments(**res)
    extra_args = megatron_args.parse_to_megatron()
    extra_args['dataset'] = 'alpaca-zh#10000'
    extra_args['template_type'] = 'default-generation'
    from swift.llm.utils.megatron_utils import forward_step, train_valid_test_datasets_provider
    from megatron.core.enums import ModelType
    from megatron.training import pretrain

    train_valid_test_datasets_provider.is_distributed = True
    patch_megatron(tokenizer)
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults=extra_args)

test_pretrain()

"""
[INFO:swift] Successfully registered `/mnt/nas2/huangjintao.hjt/work/swift/swift/llm/data/dataset_info.json`
[INFO:swift] local_repo_path: /mnt/nas2/huangjintao.hjt/.cache/modelscope/_github/megatron-lm
[INFO:swift] local_repo_path: /mnt/nas2/huangjintao.hjt/.cache/modelscope/_github/megatron-patch
/mnt/nas2/huangjintao.hjt/.cache/modelscope/_github/megatron-patch/megatron_patch/model/llava/clip_encoder.py:26: UserWarning: The cvcuda environment does not exist. Install cvcuda and use it
  warnings.warn("The cvcuda environment does not exist. Install cvcuda and use it")
/mnt/nas2/huangjintao.hjt/.cache/modelscope/_github/megatron-patch/megatron_patch/model/llava/clip_encoder.py:26: UserWarning: The cvcuda environment does not exist. Install cvcuda and use it
  warnings.warn("The cvcuda environment does not exist. Install cvcuda and use it")
using world size: 2, data-parallel size: 2, context-parallel size: 1 tensor-model-parallel size: 1, pipeline-model-parallel size: 1
WARNING: overriding default arguments for use_legacy_models:False                        with use_legacy_models:False
WARNING: Setting args.overlap_p2p_comm to False since non-interleaved schedule does not support overlapping p2p communication
accumulate and all-reduce gradients in fp32 for bfloat16 data type.
using torch.bfloat16 for parameters ...
------------------------ arguments ------------------------
  accumulate_allreduce_grads_in_fp32 .............. True
  adam_beta1 ...................................... 0.9
  adam_beta2 ...................................... 0.95
  adam_eps ........................................ 1e-08
  add_bias_linear ................................. False
  add_position_embedding .......................... True
  add_qkv_bias .................................... True
  adlr_autoresume ................................. False
  adlr_autoresume_interval ........................ 1000
  apply_layernorm_1p .............................. False
  apply_query_key_layer_scaling ................... False
  apply_residual_connection_post_layernorm ........ False
  apply_rope_fusion ............................... False
  async_save ...................................... None
  async_tensor_model_parallel_allreduce ........... True
  attention_dropout ............................... 0.0
  attention_softmax_in_fp32 ....................... False
  auto_detect_ckpt_format ......................... False
  barrier_with_L1_time ............................ True
  bert_binary_head ................................ True
  bert_embedder_type .............................. megatron
  bert_load ....................................... None
  bf16 ............................................ True
  bias_dropout_fusion ............................. True
  bias_gelu_fusion ................................ False
  bias_swiglu_fusion .............................. True
  biencoder_projection_dim ........................ 0
  biencoder_shared_query_context_model ............ False
  block_data_path ................................. None
  calculate_per_token_loss ........................ False
  check_for_nan_in_loss_and_grad .................. True
  check_weight_hash_across_dp_replicas_interval ... None
  ckpt_assume_constant_structure .................. False
  ckpt_fully_parallel_load ........................ False
  ckpt_fully_parallel_save ........................ False
  ckpt_step ....................................... None
  classes_fraction ................................ 1.0
  clip_grad ....................................... 1.0
  clone_scatter_output_in_embedding ............... True
  consumed_train_samples .......................... 0
  consumed_valid_samples .......................... 0
  context_parallel_size ........................... 1
  create_attention_mask_in_dataloader ............. True
  data_cache_path ................................. None
  data_parallel_random_init ....................... False
  data_parallel_size .............................. 2
  data_path ....................................... None
  data_per_class_fraction ......................... 1.0
  data_sharding ................................... True
  dataloader_type ................................. cyclic
  dataset ......................................... ['alpaca-zh#1000']
  ddp_average_in_collective ....................... False
  ddp_bucket_size ................................. None
  decoder_num_layers .............................. None
  decoder_seq_length .............................. None
  decoupled_lr .................................... None
  decoupled_min_lr ................................ None
  delay_grad_reduce ............................... True
  delay_param_gather .............................. False
  deprecated_use_mcore_models ..................... False
  deterministic_mode .............................. False
  dino_bottleneck_size ............................ 256
  dino_freeze_last_layer .......................... 1
  dino_head_hidden_size ........................... 2048
  dino_local_crops_number ......................... 10
  dino_local_img_size ............................. 96
  dino_norm_last_layer ............................ False
  dino_teacher_temp ............................... 0.07
  dino_warmup_teacher_temp ........................ 0.04
  dino_warmup_teacher_temp_epochs ................. 30
  disable_straggler_on_startup .................... False
  dist_ckpt_format ................................ torch_dist
  distribute_saved_activations .................... False
  distributed_backend ............................. nccl
  distributed_timeout_minutes ..................... 10
  embedding_path .................................. None
  empty_unused_memory_level ....................... 0
  enable_one_logger ............................... False
  encoder_num_layers .............................. 24
  encoder_seq_length .............................. 128
  end_weight_decay ................................ 0.1
  eod_mask_loss ................................... True
  eval_interval ................................... 200
  eval_iters ...................................... 10
  evidence_data_path .............................. None
  exit_duration_in_mins ........................... None
  exit_interval ................................... None
  exit_on_missing_checkpoint ...................... False
  exit_signal_handler ............................. False
  expert_model_parallel_size ...................... 1
  extra_vocab_size ................................ 293
  ffn_hidden_size ................................. 4864
  finetune ........................................ False
  fp16 ............................................ False
  fp16_lm_cross_entropy ........................... False
  fp32_residual_connection ........................ False
  fp8 ............................................. None
  fp8_amax_compute_algo ........................... most_recent
  fp8_amax_history_len ............................ 1
  fp8_interval .................................... 1
  fp8_margin ...................................... 0
  fp8_wgrad ....................................... True
  global_batch_size ............................... 8
  gradient_accumulation_fusion .................... True
  group_query_attention ........................... True
  head_lr_mult .................................... 1.0
  hidden_dropout .................................. 0.0
  hidden_size ..................................... 896
  hysteresis ...................................... 2
  ict_head_size ................................... None
  ict_load ........................................ None
  img_h ........................................... 224
  img_w ........................................... 224
  indexer_batch_size .............................. 128
  indexer_log_interval ............................ 1000
  inference_batch_times_seqlen_threshold .......... 512
  init_method_std ................................. 0.008
  init_method_xavier_uniform ...................... False
  initial_loss_scale .............................. 4294967296
  iter_per_epoch .................................. 1250
  kv_channels ..................................... 64
  lazy_mpu_init ................................... None
  load ............................................ /mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1-new
  local_rank ...................................... None
  log_batch_size_to_tensorboard ................... True
  log_interval .................................... 10
  log_learning_rate_to_tensorboard ................ True
  log_loss_scale_to_tensorboard ................... True
  log_memory_to_tensorboard ....................... True
  log_num_zeros_in_grad ........................... False
  log_params_norm ................................. False
  log_progress .................................... False
  log_straggler ................................... False
  log_throughput .................................. False
  log_timers_to_tensorboard ....................... True
  log_validation_ppl_to_tensorboard ............... True
  log_world_size_to_tensorboard ................... False
  logging_level ................................... None
  loss_scale ...................................... None
  loss_scale_window ............................... 1000
  lr .............................................. 1e-05
  lr_decay_iters .................................. 900
  lr_decay_samples ................................ None
  lr_decay_style .................................. cosine
  lr_warmup_fraction .............................. None
  lr_warmup_init .................................. 0.0
  lr_warmup_iters ................................. 100
  lr_warmup_samples ............................... 0
  make_vocab_size_divisible_by .................... 128
  manual_gc ....................................... False
  manual_gc_eval .................................. True
  manual_gc_interval .............................. 0
  mask_factor ..................................... 1.0
  mask_prob ....................................... 0.15
  mask_type ....................................... random
  masked_softmax_fusion ........................... True
  max_position_embeddings ......................... 131072
  max_tokens_to_oom ............................... 12000
  merge_file ...................................... None
  micro_batch_size ................................ 1
  min_loss_scale .................................. 1.0
  min_lr .......................................... 1e-06
  mmap_bin_files .................................. True
  mock_data ....................................... False
  model_type ...................................... qwen2-0_5b
  moe_aux_loss_coeff .............................. 0.01
  moe_expert_capacity_factor ...................... None
  moe_extended_tp ................................. False
  moe_grouped_gemm ................................ False
  moe_input_jitter_eps ............................ None
  moe_layer_recompute ............................. False
  moe_pad_expert_input_to_capacity ................ False
  moe_per_layer_logging ........................... False
  moe_router_load_balancing_type .................. aux_loss
  moe_router_topk ................................. 2
  moe_token_dispatcher_type ....................... allgather
  moe_token_drop_policy ........................... probs
  moe_z_loss_coeff ................................ None
  nccl_communicator_config_path ................... None
  no_load_optim ................................... None
  no_load_rng ..................................... None
  no_persist_layer_norm ........................... False
  no_save_optim ................................... None
  no_save_rng ..................................... None
  norm_epsilon .................................... 1e-06
  normalization ................................... RMSNorm
  num_attention_heads ............................. 14
  num_channels .................................... 3
  num_classes ..................................... 1000
  num_dataset_builder_threads ..................... 1
  num_experts ..................................... None
  num_layers ...................................... 24
  num_layers_per_virtual_pipeline_stage ........... None
  num_query_groups ................................ 2
  num_workers ..................................... 1
  one_logger_entity ............................... hwinf_dcm
  one_logger_project .............................. e2e-tracking
  one_logger_run_name ............................. None
  onnx_safe ....................................... None
  openai_gelu ..................................... False
  optimizer ....................................... adam
  output_bert_embeddings .......................... False
  overlap_grad_reduce ............................. False
  overlap_p2p_comm ................................ False
  overlap_param_gather ............................ False
  override_opt_param_scheduler .................... False
  params_dtype .................................... torch.bfloat16
  patch_dim ....................................... 16
  perform_initialization .......................... True
  pipeline_model_parallel_size .................... 1
  pipeline_model_parallel_split_rank .............. None
  position_embedding_type ......................... rope
  pretrained_checkpoint ........................... None
  profile ......................................... False
  profile_ranks ................................... [0]
  profile_step_end ................................ 12
  profile_step_start .............................. 10
  qk_layernorm .................................... False
  query_in_block_prob ............................. 0.1
  rampup_batch_size ............................... None
  rank ............................................ 0
  recompute_granularity ........................... selective
  recompute_method ................................ None
  recompute_num_layers ............................ None
  reset_attention_mask ............................ False
  reset_position_ids .............................. False
  retriever_report_topk_accuracies ................ []
  retriever_score_scaling ......................... False
  retriever_seq_length ............................ 256
  retro_add_retriever ............................. False
  retro_attention_gate ............................ 1
  retro_cyclic_train_iters ........................ None
  retro_encoder_attention_dropout ................. 0.1
  retro_encoder_hidden_dropout .................... 0.1
  retro_encoder_layers ............................ 2
  retro_num_neighbors ............................. 2
  retro_num_retrieved_chunks ...................... 2
  retro_project_dir ............................... None
  retro_verify_neighbor_count ..................... True
  rotary_base ..................................... 1000000
  rotary_interleaved .............................. False
  rotary_percent .................................. 1.0
  rotary_seq_len_interpolation_factor ............. 1
  sample_rate ..................................... 1.0
  save ............................................ output
  save_interval ................................... 100
  scatter_gather_tensors_in_pipeline .............. True
  seed ............................................ 1234
  seq_length ...................................... 128
  sequence_parallel ............................... False
  sgd_momentum .................................... 0.9
  short_seq_prob .................................. 0.1
  skip_train ...................................... False
  spec ............................................ None
  split ........................................... None
  squared_relu .................................... False
  standalone_embedding_stage ...................... False
  start_weight_decay .............................. 0.1
  straggler_ctrlr_port ............................ 65535
  straggler_minmax_count .......................... 1
  swiglu .......................................... True
  swin_backbone_type .............................. tiny
  target_expert_model_parallel_size ............... 1
  target_pipeline_model_parallel_size ............. 1
  target_tensor_model_parallel_size ............... 1
  template_type ................................... qwen
  tensor_model_parallel_size ...................... 1
  tensorboard_dir ................................. output/tensorboard
  tensorboard_log_interval ........................ 1
  tensorboard_queue_size .......................... 1
  test_data_path .................................. []
  test_mode ....................................... False
  timing_log_level ................................ 0
  timing_log_option ............................... minmax
  titles_data_path ................................ None
  tokenizer_model ................................. None
  tokenizer_type .................................. None
  tp_comm_bulk_dgrad .............................. True
  tp_comm_bulk_wgrad .............................. True
  tp_comm_overlap ................................. False
  tp_comm_overlap_ag .............................. True
  tp_comm_overlap_cfg ............................. None
  tp_comm_overlap_rs .............................. True
  tp_comm_overlap_rs_dgrad ........................ False
  tp_comm_split_ag ................................ True
  tp_comm_split_rs ................................ True
  train_data_path ................................. []
  train_iters ..................................... 1000
  train_samples ................................... None
  transformer_impl ................................ transformer_engine
  transformer_pipeline_model_parallel_size ........ 1
  untie_embeddings_and_output_weights ............. True
  use_checkpoint_args ............................. False
  use_checkpoint_opt_param_scheduler .............. False
  use_cpu_initialization .......................... None
  use_dist_ckpt ................................... False
  use_distributed_optimizer ....................... True
  use_flash_attn .................................. False
  use_legacy_models ............................... False
  use_one_sent_docs ............................... False
  use_ring_exchange_p2p ........................... False
  use_rotary_position_embeddings .................. True
  use_tp_pp_dp_mapping ............................ False
  valid_data_path ................................. []
  variable_seq_lengths ............................ False
  virtual_pipeline_model_parallel_size ............ None
  vision_backbone_type ............................ vit
  vision_pretraining .............................. False
  vision_pretraining_type ......................... classify
  vocab_extra_ids ................................. 0
  vocab_file ...................................... None
  vocab_size ...................................... None
  wandb_exp_name ..................................
  wandb_project ...................................
  wandb_save_dir ..................................
  weight_decay .................................... 0.1
  weight_decay_incr_style ......................... constant
  world_size ...................................... 2
  yaml_cfg ........................................ None
-------------------- end of arguments ---------------------
setting number of micro-batches to constant 4
[INFO:swift] Downloading the model from ModelScope Hub, model_id: qwen/Qwen2-7B-Instruct
[INFO:swift] Loading the model using model_dir: /mnt/nas2/huangjintao.hjt/.cache/modelscope/hub/qwen/Qwen2-7B-Instruct
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
> setting tensorboard ...
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
torch distributed is already initialized, skipping initialization ...
> initialized tensor model parallel with size 1
> initialized pipeline model parallel with size 1
> setting random seeds to 1234 ...
> compiling dataset index builder ...
make: 进入目录“/mnt/nas2/huangjintao.hjt/work/Megatron-LM/megatron/core/datasets”
make: 对“default”无需做任何事。
make: 离开目录“/mnt/nas2/huangjintao.hjt/work/Megatron-LM/megatron/core/datasets”
>>> done with dataset index builder. Compilation time: 0.026 seconds
WARNING: constraints for invoking optimized fused softmax kernel are not met. We default back to unfused kernel invocations.
> compiling and loading fused kernels ...
>>> done with compiling and loading fused kernels. Compilation time: 0.006 seconds[rank1]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())

[rank0]:[W init.cpp:767] Warning: nvfuser is no longer supported in torch script, use _jit_set_nvfuser_enabled is deprecated and a no-op (function operator())
time to initialize megatron (seconds): 71.935
[after megatron is initialized] datetime: 2024-07-15 15:24:49
 > number of parameters on (tensor, pipeline) model parallel rank (0, 0): 630167424
INFO:megatron.core.distributed.distributed_data_parallel:Setting up DistributedDataParallel with config DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False, use_distributed_optimizer=True, check_for_nan_in_grad=True, bucket_size=None, average_in_collective=False)
INFO:megatron.core.distributed.param_and_grad_buffer:Number of buckets for gradient all-reduce / reduce-scatter: 1
Params for bucket 1 (630167424 elements):
        module.decoder.layers.23.mlp.linear_fc2.weight
        module.decoder.layers.17.mlp.linear_fc2.weight
        module.decoder.layers.10.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.18.self_attention.linear_qkv.bias
        module.decoder.layers.10.self_attention.linear_proj.weight
        module.decoder.layers.5.self_attention.linear_proj.weight
        module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.15.self_attention.linear_qkv.weight
        module.decoder.layers.21.self_attention.linear_qkv.weight
        module.decoder.layers.18.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.4.mlp.linear_fc1.weight
        module.decoder.layers.13.mlp.linear_fc2.weight
        module.decoder.layers.11.mlp.linear_fc2.weight
        module.decoder.layers.6.mlp.linear_fc2.weight
        module.decoder.layers.21.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.19.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.18.self_attention.linear_qkv.weight
        module.decoder.layers.15.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.13.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.12.self_attention.linear_qkv.bias
        module.decoder.layers.7.self_attention.linear_qkv.bias
        module.decoder.layers.3.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.19.mlp.linear_fc1.weight
        module.decoder.layers.12.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.10.mlp.linear_fc1.weight
        module.decoder.layers.9.mlp.linear_fc2.weight
        module.decoder.layers.7.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.3.self_attention.linear_qkv.bias
        module.decoder.layers.3.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.16.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.22.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.19.self_attention.linear_proj.weight
        module.decoder.layers.13.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.10.self_attention.linear_qkv.bias
        module.decoder.layers.1.self_attention.linear_qkv.weight
        module.decoder.layers.0.self_attention.linear_proj.weight
        module.decoder.layers.22.mlp.linear_fc1.weight
        module.decoder.layers.21.mlp.linear_fc1.weight
        module.decoder.layers.16.mlp.linear_fc1.weight
        module.decoder.layers.15.mlp.linear_fc1.weight
        module.decoder.layers.12.self_attention.linear_qkv.weight
        module.decoder.layers.8.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.7.self_attention.linear_qkv.weight
        module.decoder.layers.22.self_attention.linear_proj.weight
        module.decoder.layers.16.self_attention.linear_proj.weight
        module.decoder.layers.13.self_attention.linear_proj.weight
        module.decoder.layers.0.mlp.linear_fc2.weight
        module.decoder.layers.0.self_attention.linear_qkv.bias
        module.decoder.layers.13.self_attention.linear_qkv.weight
        module.decoder.layers.10.self_attention.linear_qkv.weight
        module.decoder.layers.8.self_attention.linear_proj.weight
        module.decoder.layers.13.self_attention.linear_qkv.bias
        module.decoder.layers.18.mlp.linear_fc2.weight
        module.decoder.layers.5.mlp.linear_fc1.weight
        module.decoder.layers.19.self_attention.linear_qkv.bias
        module.decoder.layers.14.self_attention.linear_proj.weight
        module.decoder.layers.21.mlp.linear_fc2.weight
        module.decoder.layers.19.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.15.mlp.linear_fc2.weight
        module.decoder.layers.2.mlp.linear_fc1.weight
        module.decoder.layers.22.self_attention.linear_qkv.bias
        module.decoder.layers.16.self_attention.linear_qkv.bias
        module.decoder.layers.13.mlp.linear_fc1.weight
        module.decoder.layers.12.mlp.linear_fc2.weight
        module.decoder.layers.7.mlp.linear_fc2.weight
        module.decoder.layers.16.mlp.linear_fc1.layer_norm_weight
        module.output_layer.weight
        module.decoder.layers.22.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.20.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.19.self_attention.linear_qkv.weight
        module.decoder.layers.14.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.8.self_attention.linear_qkv.bias
        module.decoder.layers.4.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.2.mlp.linear_fc2.weight
        module.decoder.layers.2.self_attention.linear_qkv.bias
        module.decoder.layers.20.mlp.linear_fc1.weight
        module.decoder.layers.14.mlp.linear_fc1.weight
        module.decoder.layers.3.self_attention.linear_qkv.weight
        module.decoder.layers.2.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.0.mlp.linear_fc1.weight
        module.decoder.layers.23.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.22.self_attention.linear_qkv.weight
        module.decoder.layers.20.self_attention.linear_proj.weight
        module.decoder.layers.17.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.16.self_attention.linear_qkv.weight
        module.decoder.layers.4.self_attention.linear_proj.weight
        module.decoder.layers.0.self_attention.linear_qkv.weight
        module.decoder.layers.23.mlp.linear_fc1.weight
        module.decoder.layers.17.mlp.linear_fc1.weight
        module.decoder.layers.8.self_attention.linear_qkv.weight
        module.decoder.layers.5.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.5.self_attention.linear_qkv.bias
        module.decoder.layers.1.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.23.self_attention.linear_proj.weight
        module.decoder.layers.17.self_attention.linear_proj.weight
        module.decoder.layers.5.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.1.mlp.linear_fc1.weight
        module.decoder.layers.11.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.6.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.19.mlp.linear_fc2.weight
        module.decoder.layers.11.mlp.linear_fc1.weight
        module.decoder.layers.6.mlp.linear_fc1.weight
        module.decoder.layers.4.mlp.linear_fc2.weight
        module.decoder.layers.20.self_attention.linear_qkv.bias
        module.decoder.layers.14.self_attention.linear_qkv.bias
        module.decoder.layers.11.self_attention.linear_proj.weight
        module.decoder.layers.9.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.6.self_attention.linear_proj.weight
        module.decoder.layers.14.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.16.mlp.linear_fc2.weight
        module.decoder.final_layernorm.weight
        module.decoder.layers.22.mlp.linear_fc2.weight
        module.decoder.layers.20.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.9.mlp.linear_fc1.weight
        module.decoder.layers.8.mlp.linear_fc1.weight
        module.decoder.layers.2.self_attention.linear_proj.weight
        module.decoder.layers.23.self_attention.linear_qkv.bias
        module.decoder.layers.17.self_attention.linear_qkv.bias
        module.decoder.layers.9.self_attention.linear_proj.weight
        module.decoder.layers.3.mlp.linear_fc2.weight
        module.decoder.layers.23.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.21.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.20.self_attention.linear_qkv.weight
        module.decoder.layers.17.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.14.self_attention.linear_qkv.weight
        module.decoder.layers.15.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.10.mlp.linear_fc2.weight
        module.decoder.layers.5.mlp.linear_fc2.weight
        module.decoder.layers.2.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.23.self_attention.linear_qkv.weight
        module.decoder.layers.21.self_attention.linear_proj.weight
        module.decoder.layers.18.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.17.self_attention.linear_qkv.weight
        module.decoder.layers.15.self_attention.linear_proj.weight
        module.decoder.layers.11.self_attention.linear_qkv.bias
        module.decoder.layers.6.self_attention.linear_qkv.bias
        module.decoder.layers.4.self_attention.linear_qkv.weight
        module.decoder.layers.18.mlp.linear_fc1.weight
        module.decoder.layers.11.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.8.mlp.linear_fc2.weight
        module.decoder.layers.6.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.4.self_attention.linear_qkv.bias
        module.decoder.layers.1.mlp.linear_fc2.weight
        module.decoder.layers.1.self_attention.linear_proj.weight
        module.decoder.layers.18.self_attention.linear_proj.weight
        module.decoder.layers.9.self_attention.linear_qkv.bias
        module.decoder.layers.1.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.12.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.11.self_attention.linear_qkv.weight
        module.decoder.layers.9.mlp.linear_fc1.layer_norm_weight
        module.decoder.layers.7.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.6.self_attention.linear_qkv.weight
        module.decoder.layers.4.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.3.self_attention.linear_proj.weight
        module.decoder.layers.1.self_attention.linear_qkv.bias
        module.decoder.layers.14.mlp.linear_fc2.weight
        module.decoder.layers.20.mlp.linear_fc2.weight
        module.decoder.layers.12.mlp.linear_fc1.weight
        module.decoder.layers.7.mlp.linear_fc1.weight
        module.decoder.layers.3.mlp.linear_fc1.weight
        module.decoder.layers.2.self_attention.linear_qkv.weight
        module.decoder.layers.21.self_attention.linear_qkv.bias
        module.decoder.layers.15.self_attention.linear_qkv.bias
        module.decoder.layers.12.self_attention.linear_proj.weight
        module.decoder.layers.10.self_attention.linear_qkv.layer_norm_weight
        module.decoder.layers.9.self_attention.linear_qkv.weight
        module.decoder.layers.7.self_attention.linear_proj.weight
        module.decoder.layers.5.self_attention.linear_qkv.weight
        module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight
        module.embedding.word_embeddings.weight
INFO:megatron.core.optimizer:Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=1e-05, min_lr=1e-06, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.1, fp16=False, bf16=True, params_dtype=torch.bfloat16, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=True, overlap_grad_reduce=False, overlap_param_gather=False, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=True, timers=<megatron.core.timers.Timers object at 0x7f88e6abf850>)
> learning rate decay style: cosine
 loading release checkpoint from /mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1-new
 checkpoint version 3.0
  successfully loaded checkpoint from /mnt/nas2/huangjintao.hjt/work/Pai-Megatron-Patch/qwen-ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1-new [ t 0, p 0 ] at iteration 0
(min, max) time across ranks (ms):
    load-checkpoint ................................: (1267.00, 1267.01)
[after model, optimizer, and learning rate scheduler are built] datetime: 2024-07-15 15:24:50
> building train, validation, and test datasets ...
 > datasets target sizes (minimum size):
    train:      8000
    validation: 480
    test:       80
[INFO:swift] > building train, validation, and test datasets for GPT ...
[INFO:swift] Downloading the dataset from ModelScope, dataset_id: AI-ModelScope/alpaca-gpt4-data-zh
[INFO:modelscope] Context manager of ms-dataset exited.
[INFO:modelscope] Context manager of ms-dataset exited.
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████| 990/990 [00:00<00:00, 4917.08 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████| 990/990 [00:00<00:00, 4740.79 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 173.64 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 168.47 examples/s]
[after dataloaders are built] datetime: 2024-07-15 15:24:58
done with setup ...
(min, max) time across ranks (ms):
    model-and-optimizer-setup ......................: (1538.42, 1540.23)
    train/valid/test-data-iterators-setup ..........: (8140.10, 8140.19)training ...

[before the start of training step] datetime: 2024-07-15 15:24:59
Number of parameters in transformer layers in billions:  0.36
Number of parameters in embedding layers in billions: 0.27
Total number of parameters in billions: 0.63
Number of parameters in most loaded shard in billions: 0.6302
Theoretical memory footprints: weight and optimizer=7211.88 MB
[Rank 0] (after 10 iterations) memory (MB) | allocated: 7253.681640625 | max allocated: 7959.9697265625 | reserved: 8266.0 | max reserved: 8266.0
 [2024-07-15 15:26:21] iteration       10/    1000 | consumed samples:           80 | elapsed time per iteration (ms): 8195.9 | learning rate: 1.000000E-06 | global batch size:     8 | lm loss: 8.482397E+00 | loss scale: 1.0 | grad norm: 184.363 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:26:29] iteration       20/    1000 | consumed samples:          160 | elapsed time per iteration (ms): 786.2 | learning rate: 2.000000E-06 | global batch size:     8 | lm loss: 8.724027E+00 | loss scale: 1.0 | grad norm: 84.046 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:26:37] iteration       30/    1000 | consumed samples:          240 | elapsed time per iteration (ms): 779.4 | learning rate: 3.000000E-06 | global batch size:     8 | lm loss: 6.544408E+00 | loss scale: 1.0 | grad norm: 72.223 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:26:44] iteration       40/    1000 | consumed samples:          320 | elapsed time per iteration (ms): 729.9 | learning rate: 4.000000E-06 | global batch size:     8 | lm loss: 4.188241E+00 | loss scale: 1.0 | grad norm: 72.571 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:26:51] iteration       50/    1000 | consumed samples:          400 | elapsed time per iteration (ms): 705.2 | learning rate: 5.000000E-06 | global batch size:     8 | lm loss: 1.564374E+00 | loss scale: 1.0 | grad norm: 30.492 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:26:58] iteration       60/    1000 | consumed samples:          480 | elapsed time per iteration (ms): 699.0 | learning rate: 6.000000E-06 | global batch size:     8 | lm loss: 1.241556E+00 | loss scale: 1.0 | grad norm: 26.771 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:27:06] iteration       70/    1000 | consumed samples:          560 | elapsed time per iteration (ms): 717.1 | learning rate: 7.000000E-06 | global batch size:     8 | lm loss: 1.167308E+00 | loss scale: 1.0 | grad norm: 29.694 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:27:12] iteration       80/    1000 | consumed samples:          640 | elapsed time per iteration (ms): 692.3 | learning rate: 8.000000E-06 | global batch size:     8 | lm loss: 9.790138E-01 | loss scale: 1.0 | grad norm: 13.260 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:27:20] iteration       90/    1000 | consumed samples:          720 | elapsed time per iteration (ms): 728.1 | learning rate: 9.000000E-06 | global batch size:     8 | lm loss: 9.920629E-01 | loss scale: 1.0 | grad norm: 7.949 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:27:27] iteration      100/    1000 | consumed samples:          800 | elapsed time per iteration (ms): 742.5 | learning rate: 1.000000E-05 | global batch size:     8 | lm loss: 9.911994E-01 | loss scale: 1.0 | grad norm: 10.575 | number of skipped iterations:   0 | number of nan iterations:   0 |
(min, max) time across ranks (ms):
    save-checkpoint ................................: (30421.34, 30421.36)
 [2024-07-15 15:28:05] iteration      110/    1000 | consumed samples:          880 | elapsed time per iteration (ms): 719.8 | learning rate: 9.996531E-06 | global batch size:     8 | lm loss: 9.327390E-01 | loss scale: 1.0 | grad norm: 7.640 | number of skipped iterations:   0 | number of nan iterations:   0 |
 [2024-07-15 15:28:12] iteration      120/    1000 | consumed samples:          960 | elapsed time per iteration (ms): 717.2 | learning rate: 9.986128E-06 | global batch size:     8 | lm loss: 9.523046E-01 | loss scale: 1.0 | grad norm: 6.920 | number of skipped iterations:   0 | number of nan iterations:   0 |
"""
