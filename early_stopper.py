"""This module contains a class that implements early stopping regularization technique"""

class EarlyStopper:
    """Class that facilitates early stopping to prevent model overfitting

    Attributes:
        patience: the number of epochs in a row where 
                  it's ok that the validation loss is not decreasing
        best_loss: keeps track of the current lowest validation loss value
        counter: keeps track of the number of epochs in which the validation loss hasn't decreased
        save_model: if True, make a checkpoint of the model

    Methods:
        check: check whether the model should be stopped early. If not, check if a checkpoint should be made
    """

    def __init__(self, patience: int = 2):
        """Create an early stopper with patience
        
        Args: 
            patience: the number of epochs in a row where 
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