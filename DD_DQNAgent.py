import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import SumTree as st
import random
import sys
from collections import deque

class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = st.SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani

def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def conv2d_layer(input, channel_in, channel_out, kernel_size, strides, name="conv"):
    with tf.name_scope(name):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        # print(str(input.shape[0]))
        w = tf.Variable(init(shape=[kernel_size, kernel_size, channel_in, channel_out]), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[channel_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, strides, strides, 1], padding="VALID")
        act = tf.nn.elu(conv + b)
        tf.compat.v1.summary.histogram("weights", w)
        tf.compat.v1.summary.histogram("biases", b)
        tf.compat.v1.summary.histogram("activations", act)

        return act


def fc_layer(input, channel_in, channel_out, activation=True, name="fc"):
    with tf.name_scope(name):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        w = tf.Variable(init(shape=[channel_in, channel_out]), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="B")

        if activation:
            act = tf.nn.elu(tf.matmul(input, w) + b)
            # print(act)
        else:
            act = tf.matmul(input, w) + b
            # print(act)

        tf.compat.v1.summary.histogram("weights", w)
        tf.compat.v1.summary.histogram("biases", b)
        tf.compat.v1.summary.histogram("activations", act)

        return act


class Agent:
    one_hot_actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def __init__(self, state_size, action_size, learning_rate, L, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        self.K = state_size[0]
        self.L = L

        # remember dome location:
        self.dome_location = [0.0, 0.0]
        self.dome_found = False
        self.is_first_dome = True

        # remember enemy location:
        self.enemy_location = [0.0, 0.0]
        self.enemy_found = False
        self.is_first_enemy = True

        self.strikes = 0

        self.stacked_frame = deque([np.zeros([self.K, self.K]) for i in range(4)], maxlen=4)

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="inputs")

            image = tf.reshape(self.inputs_, [-1, *state_size])
            # print(image.shape)
            tf.compat.v1.summary.image('input', image, 3)

            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, action_size], name="actions_")

            self.angle = tf.compat.v1.placeholder(tf.float32, name='angle')

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """
            # Input is 50x50x2
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[8, 8],
                                          strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            # self.conv1 = conv2d_layer(self.inputs_, 1, 16, 4, 4, "conv1")
            # self.conv2 = conv2d_layer(self.conv1, 16, 32, 4, 2, "conv2")
            # self.conv3 = conv2d_layer(self.conv2, 32, 64, 4, 2, "conv3")
            # self.flatten = tf.layers.flatten(self.conv3)
            # # print(self.flatten.shape)
            # self.value_fc = fc_layer(self.flatten, 256, 256, True, "value_fc")
            # self.value = fc_layer(self.value_fc, 256, 1, False, "value")
            # self.advantage_fc = fc_layer(self.flatten, 256, 256, True, "advantage_fc")
            # self.advantage = fc_layer(self.advantage_fc, 256, 4, False, "advantage")

            # self.conv1 = conv2d_layer(self.inputs_, 1, 16, 8, 4, "conv1")
            # self.conv2 = conv2d_layer(self.conv1, 16, 32, 4, 4, "conv2")
            # self.conv3 = conv2d_layer(self.conv2, 32, 64, 2, 4, "conv3")
            # self.flatten = tf.layers.flatten(self.conv3)
            # self.value_fc = fc_layer(self.flatten, 64, 64, True, "value_fc")
            # # print(self.value_fc)
            # self.value = fc_layer(self.value_fc, 64, 1, False, "value")
            # self.advantage_fc = fc_layer(self.flatten, 64, 64, True, "advantage_fc")
            # self.advantage = fc_layer(self.advantage_fc, 64, 4, False, "advantage")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            # Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")
            # with tf.variable_scope('value_fc', reuse=tf.AUTO_REUSE):
            #     self.w = tf.get_variable('kernel')

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.math.squared_difference(self.target_Q, self.Q))

            # self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

            # d2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            # # print(d2_vars[0])
            # tf.summary.histogram("weights", d2_vars[0])
            # tf.summary.histogram("biases", d2_vars[1])

            # for var in tf.trainable_variables():
            #     # print(var.eval())
            #     tf.compat.v1.summary.histogram(var.name, var.eval())

    def predict_action(self, explore_start, explore_stop, decay_rate, decay_step, state, sess):
        """
        This function will do the part
        With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
        """
        # EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        # First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            action = random.randint(0, self.action_size - 1)

        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})
            # print(Qs)
            # Take the biggest Q value (= the best action)
            action = np.argmax(Qs)

        return action, explore_probability

    def calculate_strikes(self, score):
        if score < -50 and self.strikes == 0:
            self.strikes = 1
        elif -200 < score < -100:
            self.strikes = 2
        elif score < -200:
            self.strikes = 3

        return self.strikes

    def calculate_reward(self, _observation):
        rockets, interceptors, cities, turret_angle, score, _action = _observation
        abs_angle = np.abs(turret_angle)
        _reward = 0

        if score > 0:
            _reward += 1.0

        # for c in cities:
        #     c_x = c[0]  # + c[1] / 2
        #     c_y = 0.0
        #
        #     for r in rockets:
        #         r_x = r[0]
        #         r_y = r[1]
        #
        #         r_c_dx = np.abs(r_x - c_x)
        #         r_c_dy = np.abs(r_y - c_y)
        #         distance_r_c = (r_c_dx ** 2 + r_c_dy ** 2) ** 0.5
        #         _reward += 1 - (distance_r_c ** 0.2)

        # if score > 0:
        #     _reward += score

        # count_vision = 0
        # count_out_of_vision = 0
        #
        # if self.dome_found and _action != 1:
        #     d_x = self.dome_location[0]
        #     d_y = self.dome_location[1]
        #     # print("dome: ", d_x, d_y)
        #     for r in rockets:
        #         r_x = r[0]
        #         r_y = r[1]
        #
        #         distance_r_d = ((r_x - d_x) ** 2 + (r_y - d_y) * 2) ** 0.5
        #
        #         if r_x >= d_x:
        #             if 0 < turret_angle < 90.0:
        #                 p_theta = np.tan(np.deg2rad(90 - (abs_angle - 3)))
        #                 n_theta = np.tan(np.deg2rad(90 - (abs_angle + 3)))
        #
        #                 y_up = p_theta * (r_x - d_x)
        #                 y_down = n_theta * (r_x - d_x)
        #
        #                 if y_up > r_y > y_down:
        #                     # print("i see rocket in 0 < alpha < 90 at:", turret_angle, r_x, r_y, np.tan(np.deg2rad(90 - abs_angle)))
        #                     # if _action == 3:
        #                     #     _reward += (1 / distance_r_d) ** 0.5
        #                     # else:
        #                     #     _reward += 1.0
        #                     count_vision += 1
        #
        #             elif turret_angle == 0:
        #                 p_theta = -np.tan(np.deg2rad(90 - (abs_angle + 3)))
        #                 n_theta = np.tan(np.deg2rad(90 - (abs_angle + 3)))
        #
        #                 x_left = (r_y / p_theta) + d_x
        #                 x_right = (r_y / n_theta) + d_x
        #                 y_left = p_theta * (r_x - d_x)
        #                 y_right = n_theta * (r_x - d_x)
        #
        #                 if x_left < r_x < x_right and y_left < r_y > y_right:
        #                     # print("i see rocket in 0 at:", turret_angle, r_x, r_y, np.tan(np.deg2rad(90 - abs_angle)))
        #                     # if _action == 3:
        #                     #     _reward += (1 / distance_r_d) ** 0.5
        #                     # else:
        #                     #     _reward += 1.0
        #                     count_vision += 1
        #
        #             elif turret_angle == 90:
        #                 p_theta = np.tan(np.deg2rad(90 - (abs_angle - 3)))
        #                 n_theta = 0
        #
        #                 y_up = p_theta * (r_x - d_x)
        #                 y_down = n_theta * (r_x - d_x)
        #
        #                 if y_up > r_y > y_down:
        #                     # print("i see rocket in 90 at:", turret_angle, r_x, r_y, np.tan(np.deg2rad(90 - abs_angle)))
        #                     # if _action == 3:
        #                     #     _reward += (1 / distance_r_d) ** 0.5
        #                     # else:
        #                     #     _reward += 1.0
        #                     count_vision += 1
        #
        #             else:
        #                 # _reward -= 1.0
        #                 # print("i see nothing.")
        #                 count_out_of_vision += 1
        #
        #         elif r_x < d_x:
        #             if -90.0 < turret_angle < 0:
        #                 p_theta = -np.tan(np.deg2rad(90 - (abs_angle - 3)))
        #                 n_theta = -np.tan(np.deg2rad(90 - (abs_angle + 3)))
        #
        #                 y_up = p_theta * (r_x - d_x)
        #                 y_down = n_theta * (r_x - d_x)
        #
        #                 if y_up > r_y > y_down:
        #                     # print("i see rocket in -90 < alpha < 0 at:", turret_angle, r_x, r_y, np.tan(np.deg2rad(90 - abs_angle)))
        #                     # if _action == 3:
        #                     #     _reward += (1 / distance_r_d) ** 0.5
        #                     # else:
        #                     #     _reward += 1
        #                     count_vision += 1
        #
        #             elif turret_angle == -90:
        #                 p_theta = -np.tan(np.deg2rad(90 - (abs_angle + 3)))
        #                 n_theta = 0
        #
        #                 y_up = p_theta * (r_x - d_x)
        #                 y_down = n_theta * (r_x - d_x)
        #
        #                 if y_up > r_y > y_down:
        #                     # print("i see rocket in -90 at:", turret_angle, r_x, r_y, np.tan(np.deg2rad(90 - abs_angle)))
        #                     # _reward += (1 / distance_r_d) ** 0.5
        #                     count_vision += 1
        #
        #             else:
        #                 # _reward -= 1.0
        #                 # print("i see nothing.")
        #                 count_out_of_vision += 1
        #
        #     if len(rockets) == 0:
        #         _reward = 0
        #     else:
        #         _reward += count_vision / len(rockets)

        # if _action == 1:
        for c in cities:
            c_x = c[0]  # + c[1] / 2
            c_y = 0.0

            # for r in rockets:
            #     r_x = r[0]
            #     r_y = r[1]
            #
            #     r_c_dx = np.abs(r_x - c_x)
            #     r_c_dy = np.abs(r_y - c_y)
            #     distance_r_c = (r_c_dx ** 2 + r_c_dy ** 2) ** 0.5
            #     # _reward -=

            for i in interceptors:
                i_x = i[0]
                i_y = i[1]

                i_c_dx = np.abs(i_x - c_x)
                i_c_dy = np.abs(i_y - c_y)

                distance_i_c = (i_c_dx ** 2 + i_c_dy ** 2) ** 0.5

                for r in rockets:
                    r_x = r[0]
                    r_y = r[1]

                    i_r_dx = np.abs(r_x - i_x)
                    i_r_dy = np.abs(r_y - i_y)

                    distance_i_r = (i_r_dx ** 2 + i_r_dy ** 2) ** 0.5

                    r_c_dx = np.abs(r_x - c_x)
                    r_c_dy = np.abs(r_y - c_y)

                    distance_r_c = (r_c_dx ** 2 + r_c_dy ** 2) ** 0.5

                    cos_alpha = (distance_i_c ** 2 + distance_r_c ** 2 - distance_i_r ** 2) / \
                                (2 * distance_i_c * distance_r_c)

                    alpha = np.rad2deg(np.arccos(cos_alpha))
                    # print(alpha ** (distance_i_r ** 0.5))
                    # _reward = (1 / alpha) + (1 / distance_i_r) - (len(rockets) / distance_r_c)
                    angle_reward = 1 - (alpha / 180) ** 0.4
                    distance_reward = 1 - (distance_i_r / distance_r_c) ** 0.4
                    _reward += (angle_reward + distance_reward) / 2
                _reward *= 0.02

        # print(_reward, _action, turret_angle)
        return _reward

    def observe2(self, _observation, is_new):
        if is_new:
            return self.stack_frames(_observation, is_new)
        else:
            rockets, interceptors, cities, turret_angle, score, _action = _observation

            self.find_dome(interceptors)

            # find extreme cities
            max_city = -sys.maxsize
            min_city = sys.maxsize
            for c in cities:
                x1 = c[0] - (c[1] / 2)
                x2 = c[0] + (c[1] / 2)

                # print("c: ", c)
                if x1 < min_city:
                    min_city = x1
                elif x2 > max_city:
                    max_city = x2

            # print("max: ", max_city)
            # print("min: ", min_city)

            frame = np.zeros([self.K, self.K])

            # WE HAVE 4 ANCHOR POINTS: (x1 - L, 0), (x2 + L, 0), (x1 - L, d(x1-x2)), (x2 + L, d(x1-x2))
            dis = np.abs((max_city + self.L) - (min_city - self.L))
            _per = dis / self.K
            TOP_LEFT = [min_city - self.L, dis]  # this is our (0,0)
            TOP_RIGHT = [max_city + self.L, dis]
            BOT_LEFT = [min_city - self.L, 0.0]
            BOT_RIGHT = [max_city + self.L, 0.0]

            for c in cities:
                c_x1 = c[0] - (c[1] / 2)
                c_x2 = c[0] + (c[1] / 2)
                dx1 = np.abs(BOT_LEFT[0] - c_x1)
                dx2 = np.abs(BOT_LEFT[0] - c_x2)
                dy = 0

                x1 = int(dx1 / _per)
                x2 = int(dx2 / _per)
                y = int(dy / _per)
                # print("city x, x, y: ", x1, x2, y)
                frame[self.K - y - 1: self.K - y, x1: x2 + 1] = 1

            for r in rockets:
                r_x = r[0]
                r_y = r[1]

                # if our rocket is within the frame
                if BOT_LEFT[0] < r_x < TOP_RIGHT[0] and BOT_LEFT[1] < r_y < TOP_RIGHT[1]:
                    dx = np.abs(TOP_LEFT[0] - r_x)
                    dy = np.abs(TOP_LEFT[1] - r_y)

                    x = int(dx / _per)
                    y = int(dy / _per)
                    # print("rocket: ", x, y)
                    frame[y: y + 1, x: x + 1] += 1

            for i in interceptors:
                i_x = i[0]
                i_y = i[1]
                # print("? ", i_x, i_y)
                # print("! ", BOT_LEFT[0], TOP_RIGHT[0], TOP_RIGHT[1])
                # if our rocket is within the frame
                if BOT_LEFT[0] < i_x < TOP_RIGHT[0] and BOT_LEFT[1] < i_y < TOP_RIGHT[1]:
                    dx = np.abs(TOP_LEFT[0] - i_x)
                    dy = np.abs(TOP_LEFT[1] - i_y)

                    x = int(dx / _per)
                    y = int(dy / _per)
                    # print("i: ", x, y)
                    # frame[x - 1:x, ]
                    frame[y: y + 1, x: x + 1] += 2  # center
                    # frame[y - 1: y, x - 1: x] += 1
                    # frame[y - 1: y, x + 1: x + 2] += 1
                    # frame[y + 1: y + 2, x - 1: x] += 1
                    # frame[y + 1: y + 2, x + 1: x + 2] += 1

            if self.dome_found:
                d_x = self.dome_location[0]
                d_y = self.dome_location[1]
                # print(d_x, d_y)
                # print(BOT_LEFT[0], TOP_RIGHT[0])
                # if our dome is within the frame
                if BOT_LEFT[0] < d_x < TOP_RIGHT[0] and BOT_LEFT[1] <= d_y < TOP_RIGHT[1]:
                    dx = np.abs(TOP_LEFT[0] - d_x)
                    dy = np.abs(TOP_LEFT[1] - d_y)

                    x = int(dx / _per)
                    y = int(dy / _per)
                    # print("dome: ", x, y)
                    frame[y - 1: y, x: x + 1] += turret_angle

                    # if turret_angle == 0.0:
                    #     frame[y - 5: y, x: x+1] = 1
                    # elif turret_angle == 90.0:
                    #     frame[y - 1: y, x: x + 5] = 1
                    # elif turret_angle == -90.0:
                    #     frame[y - 1: y, x - 5: x] = 1
                    #
                    # elif 0.0 < turret_angle < 90.0:
                    #     # calculate angle to top right corner
                    #     dx = np.abs(TOP_RIGHT[0] - d_x)
                    #     dy = np.abs(TOP_RIGHT[1] - d_y)
                    #
                    #     t_x = 10 * np.sin(np.rad2deg(turret_angle))
                    #     t_y = 10 * np.cos(np.rad2deg(turret_angle))
                    #
                    #     for i in range(np.maximum(t_x, t_y)):
                    #
                    #
                    # elif -90.0 < turret_angle < 0.0:
                    #     # calculate angle to top left corner
                    #     dx = np.abs(TOP_LEFT[0] - d_x)
                    #     dy = np.abs(TOP_LEFT[1] - d_y)
                    #
                    #     alpha = np.rad2deg(np.arctan((dy / dx)))

            if self.enemy_found:
                e_x = self.enemy_location[0]
                e_y = self.enemy_location[1]
                # print(e_x, e_y)

                # if our enemy dome is within the frame
                if BOT_LEFT[0] < e_x < TOP_RIGHT[0] and BOT_LEFT[1] <= e_y < TOP_RIGHT[1]:
                    dx = np.abs(TOP_LEFT[0] - e_x)
                    dy = np.abs(TOP_LEFT[1] - e_y)

                    x = int(dx / _per)
                    y = int(dy / _per)
                    # print("enemy: ", x, y)
                    frame[y: y + 1, x: x + 1] = 1

            frame /= 255.0
            # np.set_printoptions(threshold=sys.maxsize)
            # print("frame: ", frame)
            # print("reshaped frame: ", frame.reshape([*self.state_size]))
            # return frame.reshape([*self.state_size])
            return self.stack_frames(frame, is_new)

    def observe(self, _observation, is_new):
        if is_new:
            return np.zeros([*self.state_size])
            # return self.stack_frames(_observation, is_new)
        else:
            rockets, interceptors, cities, turret_angle, score, _action = _observation

            self.find_dome(interceptors)

            # find extreme cities
            max_city = -sys.maxsize
            min_city = sys.maxsize
            for c in cities:
                x1 = c[0] - (c[1] / 2)
                x2 = c[0] + (c[1] / 2)

                # print("c: ", c)
                if x1 < min_city:
                    min_city = x1
                elif x2 > max_city:
                    max_city = x2

            # print("max: ", max_city)
            # print("min: ", min_city)

            frame = np.zeros([self.K, self.K])

            # WE HAVE 4 ANCHOR POINTS: (x1 - L, 0), (x2 + L, 0), (x1 - L, d(x1-x2)), (x2 + L, d(x1-x2))
            dis = np.abs((max_city + self.L) - (min_city - self.L))
            _per = dis / self.K
            TOP_LEFT = [min_city - self.L, dis]  # this is our (0,0)
            TOP_RIGHT = [max_city + self.L, dis]
            BOT_LEFT = [min_city - self.L, 0.0]
            BOT_RIGHT = [max_city + self.L, 0.0]

            for c in cities:
                c_x1 = c[0] - (c[1] / 2)
                c_x2 = c[0] + (c[1] / 2)
                dx1 = np.abs(BOT_LEFT[0] - c_x1)
                dx2 = np.abs(BOT_LEFT[0] - c_x2)
                dy = 0

                x1 = int(dx1 / _per)
                x2 = int(dx2 / _per)
                y = int(dy / _per)
                # print("city x, x, y: ", x1, x2, y)
                frame[self.K - y - 1: self.K - y, x1: x2 + 1] = 255.0

            for r in rockets:
                r_x = r[0]
                r_y = r[1]

                # if our rocket is within the frame
                if BOT_LEFT[0] < r_x < TOP_RIGHT[0] and BOT_LEFT[1] < r_y < TOP_RIGHT[1]:
                    dx = np.abs(TOP_LEFT[0] - r_x)
                    dy = np.abs(TOP_LEFT[1] - r_y)

                    x = int(dx / _per)
                    y = int(dy / _per)
                    # print("rocket: ", x, y)
                    frame[y: y + 1, x: x + 1] += 1

            for i in interceptors:
                i_x = i[0]
                i_y = i[1]
                # print("? ", i_x, i_y)
                # print("! ", BOT_LEFT[0], TOP_RIGHT[0], TOP_RIGHT[1])
                # if our rocket is within the frame
                if BOT_LEFT[0] < i_x < TOP_RIGHT[0] and BOT_LEFT[1] < i_y < TOP_RIGHT[1]:
                    dx = np.abs(TOP_LEFT[0] - i_x)
                    dy = np.abs(TOP_LEFT[1] - i_y)

                    x = int(dx / _per)
                    y = int(dy / _per)
                    # print("i: ", x, y)
                    # frame[x - 1:x, ]
                    frame[y: y + 1, x: x + 1] += 255.0   # center
                    # frame[y - 1: y, x - 1: x] += 1
                    # frame[y - 1: y, x + 1: x + 2] += 1
                    # frame[y + 1: y + 2, x - 1: x] += 1
                    # frame[y + 1: y + 2, x + 1: x + 2] += 1

            if self.dome_found:
                d_x = self.dome_location[0]
                d_y = self.dome_location[1]
                # print(d_x, d_y)
                # print(BOT_LEFT[0], TOP_RIGHT[0])
                # if our dome is within the frame
                if BOT_LEFT[0] < d_x < TOP_RIGHT[0] and BOT_LEFT[1] <= d_y < TOP_RIGHT[1]:
                    dx = np.abs(TOP_LEFT[0] - d_x)
                    dy = np.abs(TOP_LEFT[1] - d_y)

                    x = int(dx / _per)
                    y = int(dy / _per)
                    # print("dome: ", x, y)
                    frame[y - 1: y, x: x + 1] += turret_angle

            frame /= 255.0

            return frame.reshape([*self.state_size])
            # return self.stack_frames(frame, is_new)

    def find_dome(self, interceptors):
        # print(self.dome_found)
        if not self.dome_found and self.is_first_dome:
            # the first rocket is fired from the dome!
            if len(interceptors) == 1:
                x = [i[0] for i in interceptors]
                y = [i[1] for i in interceptors]
                self.dome_location = [x[0], y[0]]
                self.is_first_dome = False
                self.dome_found = True


    def stack_frames(self, state, is_new):
        if is_new:
            self.stacked_frame = deque([np.zeros([self.K, self.K]) for i in range(4)], maxlen=4)

            # s_frames.append(state)
            # s_frames.append(state)
            # s_frames.append(state)
            # s_frames.append(state)

            stack = np.stack(self.stacked_frame, axis=2)
            # print(stack)
        else:
            self.stacked_frame.append(state)

            stack = np.stack(self.stacked_frame, axis=2)
            # print(stack)

        # print(stack.shape)
        return stack
