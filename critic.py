from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K
from keras.utils import plot_model


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, learningRate, dropoutRate, l2Lambda):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        self.learningRate = learningRate
        self.dropoutRate = dropoutRate
        self.l2Lambda = l2Lambda
        self.initLimit = 3e-4

        self.build_model()

    def build_model(self):

        # Define states pathway.
        # Inputs.
        states = layers.Input(shape=(self.state_size,), name='states')
        # 1st hidden layer.
        net_states = layers.Dense(units=400, activity_regularizer=regularizers.l2(self.l2Lambda))(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dropout(self.dropoutRate)(net_states)
        # 2nd hidden layer.
        net_states = layers.Dense(units=300, activity_regularizer=regularizers.l2(self.l2Lambda))(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dropout(self.dropoutRate)(net_states)

        # Define actions pathway
        actions = layers.Input(shape=(self.action_size,), name='actions')
        # 1st hidden layer.
        net_actions = layers.Dense(units=300, activity_regularizer=regularizers.l2(self.l2Lambda))(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(self.dropoutRate)(net_actions)

        # Combine state and action pathways.
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(    units=1,
                                    name='q_values',
                                    activity_regularizer=regularizers.l2(self.l2Lambda),
                                    kernel_initializer=initializers.RandomUniform(minval=-self.initLimit, maxval=self.initLimit, seed=0),
                                    bias_initializer=initializers.RandomUniform(minval=0, maxval=self.initLimit, seed=0))(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Save model to image.
        plot_model(self.model, to_file='critic.png', show_shapes=True)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learningRate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
