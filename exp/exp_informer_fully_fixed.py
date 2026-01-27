import os
import time
import warnings
import numpy as np
import functools

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'informer': Informer,
            'informerstack': InformerStack,
        }

        # Determine e_layers based on model type
        if self.args.model == 'informer' or self.args.model == 'informerstack':
            e_layers = self.args.e_layers if self.args.model == 'informer' else self.args.s_layers

            # Build the base model
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.factor,
                self.args.d_model,
                self.args.n_heads,
                e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()

        # Wrap model based on configuration
        if self.args.use_fsdp and torch.cuda.is_available():
            # FSDP wrapping
            model = self._wrap_model_with_fsdp(model)
        elif self.args.use_multi_gpu and self.args.use_gpu:
            # Standard DataParallel
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _wrap_model_with_fsdp(self, model):
        """Wrap model with FSDP"""

        # Define auto-wrap policy
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=self.args.fsdp_auto_wrap_min_params
        )

        # Configure mixed precision
        mixed_precision_policy = None
        if self.args.use_amp:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        # Configure CPU offload
        cpu_offload_policy = None
        if self.args.fsdp_cpu_offload:
            cpu_offload_policy = CPUOffload(offload_params=True)

        # Get sharding strategy
        sharding_strategy = getattr(ShardingStrategy, self.args.fsdp_sharding_strategy)

        # Get backward prefetch strategy
        backward_prefetch = getattr(BackwardPrefetch, self.args.fsdp_backward_prefetch)

        # Wrap model with FSDP
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload_policy,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            backward_prefetch=backward_prefetch,
            limit_all_gathers=True,
        )

        # Apply activation checkpointing if requested
        if self.args.fsdp_activation_checkpointing:
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

            def check_fn(submodule):
                return isinstance(submodule, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer))

            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant_wrapper,
                check_fn=check_fn
            )

        return model

    def _get_data(self, flag):
        """Get dataset and dataloader - MODIFIED FOR FSDP"""
        args = self.args

        data_dict = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed != 'timeF' else 1

        # Determine batch size and shuffle behavior
        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq
        elif flag == 'pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
            freq = args.freq

        # Create dataset
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )

        # ✅ FIXED: Only rank 0 prints
        if self._should_print():
            print(flag, len(data_set))

        # Create dataloader with FSDP support
        if hasattr(args, 'use_fsdp') and args.use_fsdp:
            # FSDP mode: Use DistributedSampler
            sampler = DistributedSampler(
                data_set,
                shuffle=shuffle_flag,
                seed=args.seed if hasattr(args, 'seed') else 2021,
                drop_last=drop_last
            )

            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                sampler=sampler,  # Use sampler instead of shuffle
                num_workers=args.num_workers,
                drop_last=drop_last,
                pin_memory=True,  # Recommended for FSDP
                persistent_workers=True if args.num_workers > 0 else False
            )
        else:
            # Standard mode: Regular DataLoader
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
            )

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """Validation loop with FSDP support"""
        self.model.eval()
        total_loss = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        # Synchronize loss across all processes in FSDP
        if self.args.use_fsdp:
            loss_tensor = torch.tensor(total_loss).cuda()
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            total_loss = loss_tensor.item()

        self.model.train()
        return total_loss

    def train(self, setting):
        """Training loop with FSDP support"""
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        
        # ✅ FIXED: Only rank 0 creates directory
        if self._should_print():
            if not os.path.exists(path):
                os.makedirs(path)
        
        # Synchronize to ensure directory is created before all ranks continue
        if self.args.use_fsdp and dist.is_initialized():
            dist.barrier()

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            use_fsdp=self.args.use_fsdp if hasattr(self.args, 'use_fsdp') else False,
            global_rank=self.args.global_rank if hasattr(self.args, 'global_rank') else 0
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            # FSDP: Set epoch for DistributedSampler
            if self.args.use_fsdp and hasattr(train_loader, 'sampler'):
                train_loader.sampler.set_epoch(epoch)

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                # Print progress (only from rank 0 in FSDP mode)
                if (i + 1) % 100 == 0 and self._should_print():
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Backward pass with mixed precision
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            if self._should_print():
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            
            # ✅ FIXED: Synchronize training loss across all ranks
            if self.args.use_fsdp and dist.is_initialized():
                train_loss_tensor = torch.tensor(train_loss).cuda()
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = train_loss_tensor.item()
            
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            if self._should_print():
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if self._should_print():
                    print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self._load_checkpoint(best_model_path)

        return self.model

    def test(self, setting):
        """Test loop with FSDP support"""
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()

        preds = []
        trues = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)

        # ✅ FIXED: Only rank 0 prints
        if self._should_print():
            print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # ✅ FIXED: Only rank 0 prints
        if self._should_print():
            print('test shape:', preds.shape, trues.shape)

        # Result save (only from rank 0)
        folder_path = './results/' + setting + '/'
        
        # ✅ FIXED: Only rank 0 creates directory and saves files
        if self._should_print():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        # Synchronize to ensure directory is created before continuing
        if self.args.use_fsdp and dist.is_initialized():
            dist.barrier()

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        # ✅ FIXED: Only rank 0 prints and saves
        if self._should_print():
            print('mse:{}, mae:{}'.format(mse, mae))

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        """Prediction with FSDP support"""
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self._load_checkpoint(best_model_path)

        self.model.eval()

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # Result save (only from rank 0)
        folder_path = './results/' + setting + '/'
        
        # ✅ FIXED: Only rank 0 creates directory and saves files
        if self._should_print():
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        # Synchronize to ensure directory is created before continuing
        if self.args.use_fsdp and dist.is_initialized():
            dist.barrier()

        if self._should_print():
            np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """Process one batch of data"""
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Decoder input
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        # Encoder - Decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y

    def _load_checkpoint(self, path):
        """Load checkpoint with FSDP support"""
        if self.args.use_fsdp:
            # FSDP checkpoint loading
            with FSDP.state_dict_type(
                    self.model,
                    StateDictType.FULL_STATE_DICT,
            ):
                state_dict = torch.load(path, map_location='cpu')
                self.model.load_state_dict(state_dict)
        else:
            # Standard checkpoint loading
            state_dict = torch.load(path)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

    def _should_print(self):
        """Check if current process should print (only rank 0 in FSDP mode)"""
        if hasattr(self.args, 'use_fsdp') and self.args.use_fsdp:
            return self.args.global_rank == 0
        return True
