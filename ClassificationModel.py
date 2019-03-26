
class ClassificationModel:
    def __init__(self, model_name):
        self.model_name = model_name
        return

    def train(self):
        raise('ImplementationError: Method not implemented. Implement in derived class')
        return

    def evaluate(self):
        raise('ImplementationError: Method not implemented. Implement in derived class')
        return

    def load_model(self):
        raise('ImplementationError: Method not implemented. Implement in derived class')
        return

    def save_model(self):
        raise('ImplementationError: Method not implemented. Implement in derived class')
        return
