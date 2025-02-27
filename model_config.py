

class ModelConfig():

    def __init__(
            self, n_classes, hidden_size, window_size, window_step,
            train_epochs, device, model_save_path, write_path, dataset, windows):
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.windows = windows
        if self.windows:
            self.window_size = window_size
            self.window_step = window_step
        else:
            self.window_size = None
            self.window_step = None
        self.train_epochs = train_epochs
        self.device = device
        self.model_save_path = model_save_path
        self.write_path = write_path
        self.dataset = dataset
        self.windows = windows

    def __dict__(self):
        return {
            "epochs": self.train_epochs,
            "hidden_size": self.hidden_size,

            "window_size": self.window_size,
            "window_step": self.window_step,
            "n_classes": self.n_classes,
            "dataset": self.dataset,
            "device": self.device,
            "model_save_path": self.model_save_path,
            "write_path": self.write_path
        }