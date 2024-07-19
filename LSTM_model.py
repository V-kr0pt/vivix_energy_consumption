import torch
from sklearn.base import BaseEstimator, RegressorMixin
from utils.model_utils import Model_utils 


class LSTM_model:
    def __init__(self, model_params: dict, model_name: str = "LSTM"):
        self.model_utils = Model_utils()
        self.model = None
        self.model_name = model_name 
        self.model_params = model_params
        self.train_losses = []  # Store training losses for each epoch
           

    def create_model(self, input_size):
        hidden_layer_size = self.model_params['hidden_layer_size']
        num_layers = self.model_params['num_layers']
        output_size = self.model_params['output_size']

        # LSTM model and a fully connected layer
        # to map the output of the LSTM to the output size    
        self.model = torch.nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_layer_size, 
                                    num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_layer_size, output_size) 


    def forward_pass(self, x):
        lstm_out, _ = self.model(x)
        out = self.fc(lstm_out) 
        return out.view(-1) # to be an array and not a 1 column matrix


    def fit(self, X_train, y_train, epochs=10, batch_size=64, verbose=False):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_params['learning_rate'])

        # transform to Tensor
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()   
        # Create Dataloader
        train_loader = self.model_utils.create_Dataloader(X_train, y_train, batch_size)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                output = self.forward_pass(inputs)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.train_losses.append(avg_epoch_loss)
            
            if verbose:
                print(f'Epoch: {epoch}, Loss: {avg_epoch_loss}')
     

    def predict(self, X_test):
        # transform to Tensor
        X_test = torch.from_numpy(X_test).float()
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.forward_pass(X_test)
        
        # return the prediction as a numpy array
        return predictions.detach().numpy()
               

# To be possible to use GridSearchCV
class LSTMModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layer_size=50, num_layers=1, output_size=1, learning_rate=0.001, epochs=10, verbose=False):
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_size = output_size
        self.verbose = verbose
        self.model = LSTM_model({
            'hidden_layer_size': hidden_layer_size,
            'num_layers': num_layers,
            'learning_rate': learning_rate,
            'output_size': output_size
        })


    def fit(self, X, y):
        input_size = X.shape[1]
        self.model.create_model(input_size)
        self.model.fit(X, y, self.epochs, verbose=self.verbose)
        return self


    def predict(self, X):
        return self.model.predict(X)



