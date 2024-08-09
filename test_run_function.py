import importlib


def run_function_from_script():
    module_name = 'scripts.indi0001'
    _module = importlib.import_module('.', module_name)
    _module.training_loop()
