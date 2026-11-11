import gc; gc.collect(); torch.cuda.empty_cache()
print("allocated %.1f GB, reserved %.1f GB" % (torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30), "| P_ALL" , "P_ALL" in globals(), "| prompts", len(prompts), len(OTHER_CTX))
