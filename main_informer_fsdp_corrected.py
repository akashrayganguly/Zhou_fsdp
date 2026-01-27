#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Informer Training Script with FSDP Support for HPC
Fully Sharded Data Parallel (FSDP) implementation for distributed training

CORRECTED: Proper device management for FSDP vs DataParallel
"""

import sys
import os
import torch
import torch.distributed as dist
from datetime import timedelta

# Add project directory to path
if 'Zhou_fsdp' not in sys.path:
    sys.path.append('Zhou_fsdp')

from utils.tools import dotdict
from exp.exp_informer import Exp_Informer


def setup_distributed():
    """
    Initialize distributed training environment
    Returns: rank, world_size, local_rank
    """
    # Check if running in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # Set environment variables for PyTorch distributed
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        # Single GPU/CPU mode
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
    
    return rank, world_size, local_rank


def init_distributed_mode(args):
    """
    Initialize distributed training
    
    IMPORTANT: In FSDP mode, device is assigned based on local_rank,
    NOT from args.devices which is only for DataParallel mode.
    """
    if args.use_fsdp:
        rank, world_size, local_rank = setup_distributed()
        
        args.global_rank = rank
        args.world_size = world_size
        args.local_rank = local_rank
        
        # Initialize process group
        if not dist.is_initialized():
            # Use NCCL backend for GPU training
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=30)
            )
        
        # CRITICAL: Device is determined by local_rank, NOT args.devices
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            args.device = torch.device(f'cuda:{local_rank}')
            args.gpu = local_rank  # Set for compatibility
        else:
            args.device = torch.device('cpu')
            args.gpu = None
        
        # Synchronize
        if dist.is_initialized():
            dist.barrier()
        
        if rank == 0:
            print(f"Distributed training initialized:")
            print(f"  - World size: {world_size}")
            print(f"  - Global rank: {rank}")
            print(f"  - Local rank: {local_rank}")
            print(f"  - Device: {args.device}")
            print(f"  - Note: Device assigned by local_rank, not args.devices")
    else:
        args.global_rank = 0
        args.world_size = 1
        args.local_rank = 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_args():
    """Create and configure arguments with proper FSDP vs DataParallel handling"""
    args = dotdict()

    # ============================================================================
    # Model configuration
    # ============================================================================
    args.model = 'informer'  # Options: 'informer', 'informerstack'

    # ============================================================================
    # Data configuration
    # ============================================================================
    args.data = 'custom'
    args.root_path = './ETDataset/ETT-small/'
    args.data_path = 'nc_by_meff_multiple_mtau_ECL.csv'
    args.features = 'M'  # Options: 'M', 'S', 'MS'
    args.target = 'data9'
    args.freq = 'h'  # Options: 's', 't', 'h', 'd', 'b', 'w', 'm'
    args.checkpoints = './informer_checkpoints'
    args.cols = None  # None means use all columns
    args.inverse = False  # Whether to inverse output data

    # ============================================================================
    # Sequence lengths - FIXED: No multiplication by channels
    # ============================================================================
    args.seq_len = 96      # Input sequence length
    args.label_len = 48    # Start token length
    args.pred_len = 24     # Prediction sequence length

    # ============================================================================
    # Model parameters
    # ============================================================================
    args.enc_in = 9        # Encoder input size (number of features/channels)
    args.dec_in = 9        # Decoder input size
    args.c_out = 9         # Output size
    args.factor = 5        # ProbSparse attention factor
    args.d_model = 512     # Dimension of model
    args.n_heads = 8       # Number of heads
    args.e_layers = 2      # Number of encoder layers
    args.d_layers = 1      # Number of decoder layers
    args.s_layers = [3, 2, 1]  # Stack encoder layers (for informerstack)
    args.d_ff = 2048       # Dimension of FCN
    args.dropout = 0.05    # Dropout probability
    args.attn = 'prob'     # Attention type: 'prob' or 'full'
    args.embed = 'timeF'   # Time encoding: 'timeF', 'fixed', 'learned'
    args.activation = 'gelu'  # Activation function
    args.distil = True     # Whether to use distilling in encoder
    args.output_attention = False  # Whether to output attention
    args.mix = True        # Whether to use mix attention in decoder
    args.padding = 0       # Padding type

    # ============================================================================
    # Training parameters
    # ============================================================================
    args.batch_size = 4          # Batch size per GPU
    args.learning_rate = 0.0001  # Optimizer learning rate
    args.loss = 'mse'            # Loss function
    args.lradj = 'type1'         # Learning rate adjustment
    args.use_amp = False         # Use automatic mixed precision
    args.train_epochs = 10       # Number of training epochs
    args.patience = 3            # Early stopping patience
    
    # ============================================================================
    # Experiment settings
    # ============================================================================
    args.num_workers = 4    # DataLoader workers (0 for debugging)
    args.itr = 1            # Number of experiment iterations
    args.des = 'fsdp_exp'   # Experiment description
    args.seed = 2021        # Random seed for reproducibility

    # ============================================================================
    # GPU settings - CORRECTED EXPLANATION
    # ============================================================================
    args.use_gpu = True if torch.cuda.is_available() else False
    
    # IMPORTANT: Choose ONE mode - FSDP OR DataParallel, not both
    args.use_fsdp = True          # Enable FSDP distributed training
    args.use_multi_gpu = False     # Standard DataParallel (DON'T use with FSDP)
    
    # These parameters are ONLY used for DataParallel mode (when use_fsdp=False)
    # In FSDP mode, they are IGNORED
    args.gpu = 0                   # GPU ID for single GPU (non-FSDP, non-DataParallel)
    args.devices = '0,1,2,3'       # ONLY for DataParallel, NOT used in FSDP
    args.device_ids = None         # Will be set if using DataParallel
    
    # ============================================================================
    # FSDP configuration (ONLY used when use_fsdp=True)
    # ============================================================================
    args.fsdp_sharding_strategy = 'FULL_SHARD'  
    # Options: 
    #   - FULL_SHARD: Shard parameters, gradients, optimizer states (max memory savings)
    #   - SHARD_GRAD_OP: Shard gradients and optimizer states only
    #   - NO_SHARD: No sharding (equivalent to DDP)
    #   - HYBRID_SHARD: Shard within nodes, replicate across nodes
    
    args.fsdp_auto_wrap_min_params = 1e6  # Minimum parameters for auto-wrapping
    args.fsdp_backward_prefetch = 'BACKWARD_PRE'  # Options: BACKWARD_PRE, BACKWARD_POST
    args.fsdp_cpu_offload = False  # Whether to offload parameters to CPU
    args.fsdp_activation_checkpointing = False  # Whether to use activation checkpointing

    return args


def setup_gpu(args):
    """
    Setup GPU configuration
    
    IMPORTANT: This is only called for NON-FSDP mode.
    For FSDP, device is set in init_distributed_mode() based on local_rank.
    """
    if not args.use_fsdp:
        # Standard GPU setup (non-FSDP mode)
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

        if args.use_gpu:
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs available: {num_gpus}")

            if args.use_multi_gpu and num_gpus > 1:
                # DataParallel mode - use args.devices
                args.devices = args.devices.replace(' ', '')
                device_ids = args.devices.split(',')
                args.device_ids = [int(id_) for id_ in device_ids if int(id_) < num_gpus]

                if len(args.device_ids) == 0:
                    print("Warning: No valid GPU IDs. Using GPU 0.")
                    args.device_ids = [0]
                    args.use_multi_gpu = False

                args.gpu = args.device_ids[0]
                print(f"DataParallel mode: Using GPUs {args.device_ids}")
            else:
                # Single GPU mode
                args.gpu = 0
                args.device_ids = [0]
                args.use_multi_gpu = False

            torch.cuda.set_device(args.gpu)
            print(f"Using GPU: {args.gpu}")
        else:
            print("Using CPU")
    else:
        # FSDP mode - device already set in init_distributed_mode
        if args.global_rank == 0:
            print(f"FSDP mode: Device management handled by distributed training")
            print(f"  Each process automatically assigned to GPU based on local_rank")


def setup_data_parser(args):
    """Configure data parser"""
    data_parser = {
        'custom': {
            'data': 'nc_by_meff_multiple_mtau_ECL.csv',
            'T': 'data9',
            'M': [9, 9, 9],
            'S': [1, 1, 1],
            'MS': [9, 9, 1]
        },
        'ETTh1': {
            'data': 'ETTh1.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'ETTh2': {
            'data': 'ETTh2.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'ETTm1': {
            'data': 'ETTm1.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'ETTm2': {
            'data': 'ETTm2.csv',
            'T': 'OT',
            'M': [7, 7, 7],
            'S': [1, 1, 1],
            'MS': [7, 7, 1]
        },
        'WTH': {
            'data': 'WTH.csv',
            'T': 'WetBulbCelsius',
            'M': [12, 12, 12],
            'S': [1, 1, 1],
            'MS': [12, 12, 1]
        },
        'ECL': {
            'data': 'ECL.csv',
            'T': 'MT_320',
            'M': [321, 321, 321],
            'S': [1, 1, 1],
            'MS': [321, 321, 1]
        },
        'Solar': {
            'data': 'solar_AL.csv',
            'T': 'POWER_136',
            'M': [137, 137, 137],
            'S': [1, 1, 1],
            'MS': [137, 137, 1]
        },
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]


def print_args(args):
    """Print arguments (only from rank 0 in distributed mode)"""
    should_print = not args.use_fsdp or args.global_rank == 0
    
    if should_print:
        print("\n" + "=" * 80)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 80)
        print(f"Model: {args.model}")
        print(f"Data: {args.data} ({args.data_path})")
        print(f"Features: {args.features}")
        print(f"Sequence lengths: seq={args.seq_len}, label={args.label_len}, pred={args.pred_len}")
        print(f"Model params: d_model={args.d_model}, n_heads={args.n_heads}, layers=E{args.e_layers}/D{args.d_layers}")
        print(f"Training: epochs={args.train_epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        
        print(f"\nDevice Configuration:")
        if args.use_fsdp:
            print(f"  Mode: FSDP (Fully Sharded Data Parallel)")
            print(f"  Sharding strategy: {args.fsdp_sharding_strategy}")
            print(f"  World size: {args.world_size}")
            print(f"  Global rank: {args.global_rank}")
            print(f"  Local rank: {args.local_rank}")
            print(f"  Device: {args.device} (assigned by local_rank)")
            print(f"  CPU offload: {args.fsdp_cpu_offload}")
            print(f"  Activation checkpointing: {args.fsdp_activation_checkpointing}")
            print(f"  Note: args.devices='{args.devices}' is IGNORED in FSDP mode")
        elif args.use_multi_gpu:
            print(f"  Mode: DataParallel")
            print(f"  Devices: {args.device_ids}")
            print(f"  Primary GPU: {args.gpu}")
        else:
            print(f"  Mode: Single GPU/CPU")
            print(f"  Device: GPU {args.gpu}" if args.use_gpu else "CPU")
        
        print("=" * 80 + "\n")


def set_seed(seed):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For reproducibility on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function with FSDP support"""
    try:
        # ========================================================================
        # Setup
        # ========================================================================
        args = create_args()
        
        # Initialize distributed training if FSDP is enabled
        # IMPORTANT: This is where device gets assigned based on local_rank
        if args.use_fsdp:
            init_distributed_mode(args)
        else:
            args.global_rank = 0
            args.world_size = 1
            args.local_rank = 0
        
        # Print header (only rank 0)
        if args.global_rank == 0:
            print("\n" + "=" * 80)
            print("INFORMER TRAINING WITH FSDP")
            print("=" * 80)
        
        # Setup GPU/device (only for non-FSDP mode)
        # For FSDP, device is already set in init_distributed_mode
        setup_gpu(args)
        
        # Setup data parser
        setup_data_parser(args)
        
        # Additional configuration
        args.detail_freq = args.freq
        args.freq = args.freq[-1:]
        
        # Set random seed for reproducibility
        set_seed(args.seed)
        
        # Print configuration
        print_args(args)
        
        # ========================================================================
        # Training Loop
        # ========================================================================
        Exp = Exp_Informer

        for ii in range(args.itr):
            # Create experiment setting name
            setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                args.model, args.data, args.features,
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.attn, args.factor, args.embed, args.distil, args.mix, args.des, ii
            )

            if args.global_rank == 0:
                print(f"\n{'=' * 80}")
                print(f"ITERATION {ii + 1}/{args.itr}")
                print(f"Setting: {setting}")
                print(f"{'=' * 80}\n")

            # Create experiment
            exp = Exp(args)

            # Training
            if args.global_rank == 0:
                print(f'\n>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            # Testing
            if args.global_rank == 0:
                print(f'\n>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Synchronize after each iteration
            if args.use_fsdp and dist.is_initialized():
                dist.barrier()

        # ========================================================================
        # Cleanup
        # ========================================================================
        if args.global_rank == 0:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 80 + "\n")
        
        # Cleanup distributed training
        if args.use_fsdp:
            cleanup_distributed()

    except Exception as e:
        if 'args' in locals() and args.use_fsdp:
            if args.global_rank == 0:
                print(f"\nERROR: {str(e)}")
                import traceback
                traceback.print_exc()
            cleanup_distributed()
        else:
            print(f"\nERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
