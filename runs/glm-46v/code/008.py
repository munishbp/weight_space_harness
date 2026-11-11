layer = model.get_submodule(LAY["layers"][30]); print(layer)
import inspect; src = inspect.getsource(type(layer).forward); print(src[-1400:])
