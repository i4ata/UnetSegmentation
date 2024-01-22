class EarlyStopper:

    def __init__(self, patience: int = 2):
        """Create an early stopper with patience
        
        Args: 
            patience: the number of epochs in arrow where 
                      it's ok that the validation loss is not decreasing.
        """
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.save_model = False

    def check(self, validation_loss: float) -> bool:
        """Check if training should be stopped.
        The attribute `save_model` determines whether the model should be saved

        Args:
            validation_loss: current validation loss to examine
        Returns:
            bool: whether to stop training or not
        

        """
        self.save_model = False
        if validation_loss > self.best_loss:
            self.counter += 1
            if self.counter == self.patience:
                return True
        else:
            self.best_loss = validation_loss
            self.counter = 0
            self.save_model = True
        return False