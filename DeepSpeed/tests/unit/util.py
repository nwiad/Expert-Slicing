# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import torch
import deepspeed
from deepspeed.git_version_info import torch_info
from packaging import version as pkg_version


def skip_on_arch(min_arch=7):
    if deepspeed.accelerator.get_accelerator().device_name() == 'cuda':
        if torch.cuda.get_device_capability()[0] < min_arch:  #ignore-cuda
            pytest.skip(f"needs higher compute capability than {min_arch}")
    else:
        assert deepspeed.accelerator.get_accelerator().device_name() == 'xpu'
        return


def skip_on_cuda(valid_cuda):
    split_version = lambda x: map(int, x.split('.')[:2])
    if deepspeed.accelerator.get_accelerator().device_name() == 'cuda':
        CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])
        CUDA_VERSION = (CUDA_MAJOR * 10) + CUDA_MINOR
        if valid_cuda.count(CUDA_VERSION) == 0:
            pytest.skip(f"requires cuda versions {valid_cuda}")
    else:
        assert deepspeed.accelerator.get_accelerator().device_name() == 'xpu'
        return


def bf16_required_version_check(accelerator_check=True):
    split_version = lambda x: map(int, x.split('.')[:2])
    TORCH_MAJOR, TORCH_MINOR = split_version(torch_info['version'])
    NCCL_MAJOR, NCCL_MINOR = split_version(torch_info['nccl_version'])
    CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])

    # Sometimes bf16 tests are runnable even if not natively supported by accelerator
    if accelerator_check:
        accelerator_pass = torch_info['bf16_support']
    else:
        accelerator_pass = True

    if (TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)) and (CUDA_MAJOR >= 11) and (
            NCCL_MAJOR > 2 or (NCCL_MAJOR == 2 and NCCL_MINOR >= 10)) and accelerator_pass:
        return True
    else:
        return False


def required_torch_version(min_version=None, max_version=None):
    assert min_version or max_version, "Must provide a min_version or max_version argument"

    torch_version = pkg_version.parse(torch.__version__)

    if min_version and pkg_version.parse(str(min_version)) > torch_version:
        return False

    if max_version and pkg_version.parse(str(max_version)) < torch_version:
        return False

    return True


def required_amp_check():
    from importlib.util import find_spec
    if find_spec('apex') is None:
        return False
    else:
        return True
