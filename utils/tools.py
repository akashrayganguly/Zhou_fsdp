import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


class StandardScaler():
    """Standard Scaler for data normalization"""

    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean


class EarlyStopping:
    """
    Early stopping with FSDP support

    In FSDP mode, ensures all processes agree on early stopping decision
    and only rank 0 saves checkpoints.
    """

    def __init__(self, patience=7, verbose=False, delta=0, use_fsdp=False, global_rank=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            use_fsdp (bool): Whether FSDP is being used
            global_rank (int): Global rank of current process (for FSDP)
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.use_fsdp = use_fsdp
        self.global_rank = global_rank

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and (not self.use_fsdp or self.global_rank == 0):
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

        # Synchronize early_stop decision across all processes in FSDP
        if self.use_fsdp and dist.is_initialized():
            # Convert to tensor for all_reduce
            early_stop_tensor = torch.tensor(int(self.early_stop)).cuda()
            dist.all_reduce(early_stop_tensor, op=dist.ReduceOp.MAX)
            self.early_stop = bool(early_stop_tensor.item())

    def save_checkpoint(self, val_loss, model, path):
        """Saves model when validation loss decrease."""
        if self.verbose and (not self.use_fsdp or self.global_rank == 0):
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # Save checkpoint based on model type
        if self.use_fsdp:
            # FSDP checkpoint saving - only rank 0 saves
            if self.global_rank == 0:
                with FSDP.state_dict_type(
                        model,
                        StateDictType.FULL_STATE_DICT,
                        FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                ):
                    state_dict = model.state_dict()
                    torch.save(state_dict, path + '/' + 'checkpoint.pth')
        else:
            # Standard checkpoint saving
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), path + '/' + 'checkpoint.pth')
            else:
                torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')

        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust learning rate based on epoch

    Works for both standard and FSDP training (no special handling needed)
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        # Default: no adjustment
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # Only print from rank 0 in FSDP mode
        should_print = not hasattr(args, 'use_fsdp') or not args.use_fsdp or args.global_rank == 0
        if should_print:
            print('Updating learning rate to {}'.format(lr))


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization

    Works for both standard and FSDP training (operates on numpy arrays)
    In FSDP mode, this should only be called from rank 0
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with='NaN',
                             value_column_name="series_value"):
    """
    Convert TSF format to pandas dataframe

    This is a data preprocessing function and works the same for both
    standard and FSDP training
    """
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if len(line_content) != 3:  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(line_content[1])
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(line_content[1])

                    else:
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception("Missing attribute section.")

                    if not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception("A given series should contains a set of comma separated numeric values.")

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                if replace_missing_vals_with == 'NaN':
                                    numeric_series.append(np.nan)
                                elif replace_missing_vals_with == 'zero':
                                    numeric_series.append(0)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(np.nan) == len(numeric_series):
                            raise Exception("All series values are missing.")

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], '%Y-%m-%d %H-%M-%S')
                            else:
                                raise Exception("Invalid attribute type.")

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length


def save_checkpoint_fsdp(model, optimizer, epoch, path, args):
    """
    Enhanced checkpoint saving with optimizer state

    Saves both model and optimizer state for FSDP training
    Only rank 0 saves in FSDP mode
    """
    should_save = not hasattr(args, 'use_fsdp') or not args.use_fsdp or args.global_rank == 0

    if not should_save:
        return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': None,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if hasattr(args, 'use_fsdp') and args.use_fsdp:
        # FSDP checkpoint saving
        with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            checkpoint['model_state_dict'] = model.state_dict()
    else:
        # Standard checkpoint saving
        if isinstance(model, torch.nn.DataParallel):
            checkpoint['model_state_dict'] = model.module.state_dict()
        else:
            checkpoint['model_state_dict'] = model.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint_fsdp(model, optimizer, path, args):
    """
    Enhanced checkpoint loading with optimizer state

    Loads both model and optimizer state for FSDP training
    """
    checkpoint = torch.load(path, map_location='cpu')

    if hasattr(args, 'use_fsdp') and args.use_fsdp:
        # FSDP checkpoint loading
        with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
        ):
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Standard checkpoint loading
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint.get('epoch', 0)


def clip_grad_norm_fsdp(model, max_norm, args):
    """
    Gradient clipping with FSDP support

    FSDP has built-in gradient clipping that handles sharded gradients correctly
    """
    if hasattr(args, 'use_fsdp') and args.use_fsdp:
        # FSDP has its own clip_grad_norm_ method
        model.clip_grad_norm_(max_norm)
    else:
        # Standard gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def get_parameter_count(model, args):
    """
    Get total parameter count

    In FSDP mode, returns the full model parameter count (not just local shard)
    """
    if hasattr(args, 'use_fsdp') and args.use_fsdp:
        # For FSDP, we need to sum across all shards
        total_params = sum(p.numel() for p in model.parameters())

        if dist.is_initialized():
            # Sum across all processes
            param_tensor = torch.tensor(total_params).cuda()
            dist.all_reduce(param_tensor, op=dist.ReduceOp.SUM)
            # Each process has the same count, so divide by world size
            total_params = param_tensor.item() // dist.get_world_size()

        return total_params
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model, args):
    """
    Print model summary (only from rank 0 in FSDP mode)
    """
    should_print = not hasattr(args, 'use_fsdp') or not args.use_fsdp or args.global_rank == 0

    if not should_print:
        return

    total_params = get_parameter_count(model, args)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('=' * 50)
    print('Model Summary')
    print('=' * 50)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print('=' * 50)
