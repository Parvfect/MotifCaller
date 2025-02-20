

class ModelConfig():

    def __init__(
            self, n_classes, hidden_size, window_size, window_step, train_epochs,
            device, model_save_path, write_path):
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.window_step = window_step
        self.train_epochs = train_epochs
        self.device = device
        self.model_save_path = model_save_path
        self.write_path = write_path