import argparse
import numpy as np
import torch


def check_attributes(object_, attributes):

    missing = []
    for attr in attributes:
        if not hasattr(object_, attr):
            missing.append(attr)
    if len(missing) > 0:
        return False
    else:
        return True


def set_seeds(seed, cuda=True):
    if not hasattr(seed, "__iter__"):
        seed = (seed, seed, seed)
    np.random.seed(seed[0])
    torch.manual_seed(seed[1])
    if cuda: torch.cuda.manual_seed_all(seed[2])


def make_onehot(array, labels=None, axis=1, newaxis=False):

    # get labels if necessary
    if labels is None:
        labels = np.unique(array)
        labels = list(map(lambda x: x.item(), labels))

    # get target shape
    new_shape = list(array.shape)
    if newaxis:
        new_shape.insert(axis, len(labels))
    else:
        new_shape[axis] = new_shape[axis] * len(labels)

    # make zero array
    if type(array) == np.ndarray:
        new_array = np.zeros(new_shape, dtype=array.dtype)
    elif torch.is_tensor(array):
        new_array = torch.zeros(new_shape, dtype=array.dtype, device=array.device)
    else:
        raise TypeError("Onehot conversion undefined for object of type {}".format(type(array)))

    # fill new array
    n_seg_channels = 1 if newaxis else array.shape[axis]
    for seg_channel in range(n_seg_channels):
        for l, label in enumerate(labels):
            new_slc = [slice(None), ] * len(new_shape)
            slc = [slice(None), ] * len(array.shape)
            new_slc[axis] = seg_channel * len(labels) + l
            if not newaxis:
                slc[axis] = seg_channel
            new_array[tuple(new_slc)] = array[tuple(slc)] == label

    return new_array


def match_to(x, ref, keep_axes=(1,)):

    target_shape = list(ref.shape)
    for i in keep_axes:
        target_shape[i] = x.shape[i]
    target_shape = tuple(target_shape)
    if x.shape == target_shape:
        pass
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        while x.dim() < len(target_shape):
            x = x.unsqueeze(-1)

    x = x.expand(*target_shape)
    x = x.to(device=ref.device, dtype=ref.dtype)

    return x


def make_slices(original_shape, patch_shape):

    working_shape = original_shape[-len(patch_shape):]
    splits = []
    for i in range(len(working_shape)):
        splits.append([])
        for j in range(working_shape[i] // patch_shape[i]):
            splits[i].append(slice(j*patch_shape[i], (j+1)*patch_shape[i]))
        rest = working_shape[i] % patch_shape[i]
        if rest > 0:
            splits[i].append(slice((j+1)*patch_shape[i], (j+1)*patch_shape[i] + rest))

    # now we have all slices for the individual dimensions
    # we need their combinatorial combinations
    slices = list(itertools.product(*splits))
    for i in range(len(slices)):
        slices[i] = [slice(None), ] * (len(original_shape) - len(patch_shape)) + list(slices[i])

    return slices


def coordinate_grid_samples(mean, std, factor_std=5, scale_std=1.):

    relative = np.linspace(-scale_std*factor_std, scale_std*factor_std, 2*factor_std+1)
    positions = np.array([mean + i * std for i in relative]).T
    axes = np.meshgrid(*positions)
    axes = map(lambda x: list(x.ravel()), axes)
    samples = list(zip(*axes))
    samples = list(map(np.array, samples))

    return samples


def get_default_experiment_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=str, help="Working directory for experiment.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a config file.")
    parser.add_argument("-v", "--visdomlogger", action="store_true", help="Use visdomlogger.")
    parser.add_argument("-tx", "--tensorboardxlogger", type=str, default=None)
    parser.add_argument("-tl", "--telegramlogger", action="store_true")
    parser.add_argument("-dc", "--default_config", type=str, default="DEFAULTS", help="Legacy option retained for CLI compatibility.")
    parser.add_argument("-ad", "--automatic_description", action="store_true")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("-irc", "--ignore_resume_config", action="store_true", help="Legacy option retained for CLI compatibility.")
    parser.add_argument("-test", "--test", action="store_true", help="Run test instead of training")
    parser.add_argument("-g", "--grid", type=str, help="Path to a config for grid search")
    parser.add_argument("-s", "--skip_existing", action="store_true", help="Skip configs for which an experiment exists, only for grid search")
    parser.add_argument("-m", "--mods", type=str, nargs="+", default=None, help="Legacy option retained for CLI compatibility.")
    parser.add_argument("-ct", "--copy_test", action="store_true", help="Copy test files to original experiment.")

    return parser


def run_experiment(experiment, configs, args, mods=None, **kwargs):
    raise RuntimeError(
        "The legacy experiment runner has been removed from this project. "
        "Use the plain PyTorch scripts in scripts/ instead."
    )
