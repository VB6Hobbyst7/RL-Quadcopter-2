from keras import layers, models, optimizers, initializers
from keras import backend as K
from keras.utils import plot_model

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, learningRate, dropoutRate):

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.learningRate = learningRate
        self.dropoutRate = dropoutRate
        self.initLimit = 3e-3

        self.build_model()

    def build_model(self):

        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # First hidden layer with batch normalization and dropout.
        net = layers.Dense(units=400)(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dropout(self.dropoutRate)(net)

        # Second hidden layer with batch normalization and dropout.
        net = layers.Dense(units=300)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dropout(self.dropoutRate)(net)


        # Add final output layer with sigmoid activation
        net = layers.Dense( units=self.action_size,
                            kernel_initializer=initializers.RandomUniform(minval=-self.initLimit, maxval=self.initLimit, seed=0),
                            bias_initializer=initializers.RandomUniform(minval=0, maxval=self.initLimit, seed=0))(net)
        net = layers.BatchNormalization()(net)
        raw_actions = layers.Activation('tanh', name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (((x+1)/2) * self.action_range) + self.action_low, name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Save model to image.
        plot_model(self.model, to_file='actor.png', show_shapes=True)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.learningRate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
