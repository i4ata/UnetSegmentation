"""
Simple early stopping implementation
"""

class EarlyStopper:

    def __init__(self, patience: int) -> None:
        """
        Define a simple early stopper. 
        It is used to keep track of the model's performance on unseen data during training. 
        Inevitably, the model will start overfitting, i.e. the validation loss will start increasing. 
        The early stopper will detect that point in time and will indicate to terminate the training.
        
        :param patience (int): The maximum number of epochs the stopper before the validation loss has improved
        """

        self.patience = patience
        self.best_score = float('inf')
        self.counter = 0
        self.save_model = False

    def check(self, validation_score: float) -> bool:
        """
        Check whether to stop training due to suspected overfitting. 
        If the attribute 'save_model' is True after this method is called, make a new checkpoint of the model

        :param validation_score (float): The current validation score as a single real number. The goal is to minimize it

        :return stop (bool): If True, stop training, else, continue
        """
        self.save_model = False
        if validation_score > self.best_score:
            self.counter += 1
            if self.counter == self.patience:
                return True
        else:
            self.best_score = validation_score
            self.counter = 0
            self.save_model = True
        return False