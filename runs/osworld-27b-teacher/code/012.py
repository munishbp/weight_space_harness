# Free the host memory of part 1 before the virtual machines start: the prompts hold full-size screenshots.
import ctypes
for name in ("all_hard", "all_easy", "fit_hard", "fit_easy", "read_hard", "read_easy", "protect_prompts", "keys_all", "keep_all", "keys", "keep"):
    globals().pop(name, None)
gc.collect(); torch.cuda.empty_cache(); ctypes.CDLL("libc.so.6").malloc_trim(0)
print("host memory of the kernel, GB:", round(int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**30, 1), "| GPU GB:", round(torch.cuda.memory_allocated() / 2**30, 1))
