import os


class BaseProfiler():
    def __init__(self, args):
        self.args = args
        self.time_path = None
        self.mem_path = None

    def set_path(self, path):
        self.path = path

    def set_model_name(self, name):
        self.model_name = name

    def memory_profiling_path(self):
        """Get memory profiling path

        Returns:
            str: Path to memory profiling config file
        """
        if self.mem_path is not None:
            return self.mem_path
        assert self.model_name is not None, "Should specify the model name!"
        args = self.args
        memory_config_path = "configs/memory_profiling_%s_%s.json" % (args.mixed_precision, self.model_name)
        self.mem_path = os.path.join(self.path, memory_config_path)
        return self.mem_path

    def time_profiling_path(self):
        """Get time profiling path

        Returns:
            str: Path to time profiling config file
        """
        if self.time_path is not None:
            return self.time_path
        assert self.model_name is not None, "Should specify the model name!"
        args = self.args
        time_config_path = "configs/computation_profiling_%s_%s.json" % (args.mixed_precision, self.model_name)
        self.time_path = os.path.join(self.path, time_config_path)
        return self.time_path
