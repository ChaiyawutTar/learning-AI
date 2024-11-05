
class Logger():
    def __init__(self, device):
        self.device = device
        self.training_loss = 0
        self.training_step = 0

    def log_step(self, loss):
        if self.training_step % 5 == 0:
            print(f"Training at step {self.training_step}, loss: {loss}")
        self.training_step += 1