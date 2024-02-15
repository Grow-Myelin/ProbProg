import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from jax.random import PRNGKey
from parsePitchData import clean_data
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class RNNPitchPredictorModel(nn.Module):
    num_outputs: int  # Number of unique pitches
    hidden_size: int  # Size of the hidden layer in LSTM

    @nn.compact
    def __call__(self, inputs):
        # Initialize LSTM cell with hidden_size. Pass 'features' argument correctly.
        lstm_cell = nn.LSTMCell(features=self.hidden_size)

        # Initialize the hidden state and cell state of LSTM
        batch_size = inputs.shape[0]
        init_hidden = jnp.zeros((batch_size, self.hidden_size))
        init_cell = jnp.zeros((batch_size, self.hidden_size))
        lstm_state = (init_hidden, init_cell)

        # Process inputs through LSTM one step at a time
        outputs = []
        for i in range(inputs.shape[1]):  # Iterate through the sequence length
            lstm_state, output = lstm_cell(lstm_state, inputs[:, i, :])
            outputs.append(output)

        # You might want to process the outputs further, depending on your task.
        # For simplicity, let's take the output from the last time step.
        last_output = outputs[-1]

        # Pass the last output through a Dense layer for predictions
        logits = nn.Dense(self.num_outputs)(last_output)
        return logits


    
def create_train_state(rng, learning_rate, num_outputs, hidden_size, num_features):
    sequence_length = 3  # This should match your actual sequence length
    model = RNNPitchPredictorModel(num_outputs=num_outputs, hidden_size=hidden_size)
    params = model.init(rng, jnp.ones((1, sequence_length, num_features)))['params']
    tx = optax.adam(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)



def cross_entropy_loss(logits, labels):
    return -jnp.sum(labels * nn.log_softmax(logits), axis=-1).mean()

# def weighted_cross_entropy_loss(logits, labels, class_weights):
#     # logits: output from the model (before softmax)
#     # labels: true labels, one-hot encoded
#     # class_weights: array containing the weight for each class
    
#     # Compute softmax cross entropy loss
#     loss = -jnp.sum(labels * nn.log_softmax(logits), axis=-1)
    
#     # Determine the class index for each label
#     class_indices = jnp.argmax(labels, axis=-1)
    
#     # Gather the weights for each class in the batch
#     weights_for_batch = jnp.take(class_weights, class_indices)
    
#     # Weight the loss
#     weighted_loss = loss * weights_for_batch
    
#     # Return the mean loss
#     return jnp.mean(weighted_loss)


def compute_accuracy(logits, labels):
    return jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['inputs'])
        loss = cross_entropy_loss(logits, batch['targets'])
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = compute_accuracy(logits, batch['targets'])
    return state, loss, accuracy
# @jax.jit
# def train_step(state, batch, class_weights):
#     def loss_fn(params):
#         logits = state.apply_fn({'params': params}, batch['inputs'])
#         loss = weighted_cross_entropy_loss(logits, batch['targets'], class_weights)
#         return loss, logits
#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     (loss, logits), grads = grad_fn(state.params)
#     state = state.apply_gradients(grads=grads)
#     accuracy = compute_accuracy(logits, batch['targets'])
#     return state, loss, accuracy


# # Example training loop
def train_model(num_epochs, batch_size, learning_rate, num_outputs, train_data,n):
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, learning_rate, num_outputs,hidden_size=128,num_features=n)

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in train_data:  # Assuming train_data is an iterable of batches
            state, loss, accuracy = train_step(state, batch)
            epoch_loss += loss
            epoch_accuracy += accuracy
        epoch_loss /= len(train_data)
        epoch_accuracy /= len(train_data)
        print(f'Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')
# def train_model(num_epochs, batch_size, learning_rate, num_outputs, train_data, class_weights):
#     rng = jax.random.PRNGKey(0)
#     rng, init_rng = jax.random.split(rng)
#     state = create_train_state(init_rng, learning_rate, num_outputs)

#     for epoch in range(num_epochs):
#         epoch_loss = 0
#         epoch_accuracy = 0
#         for batch in train_data:  # Assuming train_data is an iterable of batches
#             state, loss, accuracy = train_step(state, batch, class_weights)
#             epoch_loss += loss
#             epoch_accuracy += accuracy
#         epoch_loss /= len(train_data)
#         epoch_accuracy /= len(train_data)
#         print(f'Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')

# Assuming you have prepared `train_data` appropriately
# train_model(num_epochs=10, batch_size=32, learning_rate=0.001, num_outputs=num_unique_pitches, train_data=prepared_train_data)

def main():
    inputs, outputs = clean_data()  # Assuming this returns correctly shaped data
    n = outputs.shape[1]
    learning_rate = 0.001
    num_epochs = 10
    # Placeholder class weights calculation
    # You should calculate these based on your dataset's specific class distribution

    # Splitting data into training and testing sets
    num_training = int(0.8 * len(inputs))
    train_inputs, test_inputs = inputs[:num_training], inputs[num_training:]
    train_outputs, test_outputs = outputs[:num_training], outputs[num_training:]

    # Initialize the model and training state
    rng = PRNGKey(0)
    state = create_train_state(rng, learning_rate,n,hidden_size=128,num_features=n)

    # Adjusted training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = len(train_inputs)  # Assuming batch size of 1 for simplicity
        for i in range(num_batches):
            inputs = train_inputs[i].reshape(1, 4, n)
            outputs = train_outputs[i].reshape(1, n)
            state, loss, accuracy = train_step(state, {'inputs': inputs, 'targets': outputs})
            epoch_loss += loss
            epoch_accuracy += accuracy
        epoch_loss /= num_batches
        epoch_accuracy /= num_batches
        print(f"Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")


    # The rest of your code remains unchanged up to the prediction part...
    
    # No need to reinitialize the model if we are using the trained state
    # model = PitchPredictorModel(num_outputs=n)  # Remove or comment out this line

    # Initialize a list to store prediction logits
    test_pred_logits = []

    # Predict in a loop
    for i in range(len(test_inputs)):
        # Reshape the input correctly as the model expects
        test_input_reshaped = test_inputs[i].reshape(1, 4, n)
        # Correctly use the trained parameters for prediction
        logits = state.apply_fn({'params': state.params}, test_input_reshaped)
        test_pred_logits.append(logits)

    # Assuming test_pred_logits is now a list of arrays, stack them
    test_pred_logits = jnp.vstack(test_pred_logits)  # Stack logits for further processing

    # Convert logits to predicted class indices
    test_pred_labels = jnp.argmax(test_pred_logits, axis=-1)

    # Ensure true_labels is correctly prepared from test_outputs
    true_labels = jnp.argmax(test_outputs, axis=-1)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, test_pred_labels)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('plots/prediction_resultsRNN.png')
if __name__ == "__main__":
    main()