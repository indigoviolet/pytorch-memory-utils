import datetime
import gc
import inspect
from dataclasses import dataclass, field
from typing import Callable

import humanize

import numpy as np
import torch
from loguru import logger

dtype_memory_size_dict = {
    torch.float64: 64 / 8,
    torch.double: 64 / 8,
    torch.float32: 32 / 8,
    torch.float: 32 / 8,
    torch.float16: 16 / 8,
    torch.half: 16 / 8,
    torch.int64: 64 / 8,
    torch.long: 64 / 8,
    torch.int32: 32 / 8,
    torch.int: 32 / 8,
    torch.int16: 16 / 8,
    torch.short: 16 / 6,
    torch.uint8: 8 / 8,
    torch.int8: 8 / 8,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16 / 8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = (
        8 / 8
    )  # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571


def get_mem_space(x):
    try:
        return dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
        raise


@dataclass
class MemTracker:
    """
    Class used to track pytorch memory usage
    Arguments:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """

    log_fn: Callable[[str], None] = logger.info
    print_detail = True
    verbose = False
    device: torch.device = torch.device("cuda:0")
    begin: bool = field(init=False, default=True)
    last_tensor_sizes: set = field(init=False, default_factory=set)

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (
                    hasattr(obj, "data") and torch.is_tensor(obj.data)
                ):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print("A trivial exception occured: {}".format(e))

    def get_tensor_usage(self):
        sizes = [
            np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype)
            for tensor in self.get_tensors()
        ]
        return self._mformat(np.sum(sizes))

    def get_allocate_usage(self):
        return self._mformat(torch.cuda.memory_allocated())

    def get_reserved_usage(self):
        return f"{self._mformat(torch.cuda.memory_reserved())} max={self._mformat(torch.cuda.max_memory_reserved())}"

    def _mformat(self, x):
        return humanize.naturalsize(x, binary=True)

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def all_gpu_tensors_info(self):
        for x in self.get_tensors():
            yield (
                x.size(),
                x.dtype,
                self._mformat(np.prod(np.array(x.size())) * get_mem_space(x.dtype)),
            )

    def _log_summary(self):
        self.log_fn(
            f"Tensor:{self.get_tensor_usage()} | Allocated:{self.get_allocate_usage()} | Reserved:{self.get_reserved_usage()}"
        )

    def _log_delimiter(self, notes: str = ""):
        frameinfo = inspect.stack()[2]
        where_str = (
            f"{frameinfo.filename} line {str(frameinfo.lineno)}: {frameinfo.function}"
        )
        self.log_fn(f"\n========== At {where_str:<50} [{notes}]")

    def track(self, notes: str = ""):
        """
        Track the GPU memory usage
        """

        self._log_delimiter(notes)

        if self.begin:
            self._log_summary()
            self.begin = False

        if self.print_detail is True:
            ts_list = [(tensor.size(), tensor.dtype) for tensor in self.get_tensors()]
            new_tensor_sizes = {
                (
                    type(x),
                    tuple(x.size()),
                    ts_list.count((x.size(), x.dtype)),
                    (np.prod(np.array(x.size())) * get_mem_space(x.dtype)),
                    x.dtype,
                )
                for x in self.get_tensors()
            }
            for t, s, n, m, data_type in new_tensor_sizes - self.last_tensor_sizes:
                self.log_fn(
                    f"+ | {str(n)} * Shape:{str(s):<20} | Memory: {self._mformat(m*n)}  | {str(t):<20} | {data_type}"
                )
            for t, s, n, m, data_type in self.last_tensor_sizes - new_tensor_sizes:
                self.log_fn(
                    f"- | {str(n)} * Shape:{str(s):<20} | Memory: {self._mformat(m*n)}  | {str(t):<20} | {data_type}"
                )

            self.last_tensor_sizes = new_tensor_sizes

        self._log_summary()
