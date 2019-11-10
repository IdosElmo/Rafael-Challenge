# import Interceptor_V2 as Interceptor_V2
from Interceptor_V2 import Init, Game_step, Draw
import tensorflow as tf
import random
import numpy as np
import DD_DQNAgent as Agent
import sys
import time

# MODEL HYPERPARAMETERS
K = 80
L = 8000    # ADD EXTRA L LENGTH TO POSITION CALCULATION
AUX_SPACE = 0  # ADDING # EXTRA FEATURES - CHECK OBSERVE FUNCTION
SHAPE = (K * K) + AUX_SPACE  # OBSERVATION SPACE K*K + M
DISTANCE_PER_GRID_SQUARE = 150
state_size = [K, K, 1]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = 4  # 4 possible actions
one_hot_actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
learning_rate = 0.0025  # Alpha (aka learning rate)

# TRAINING HYPERPARAMETERS
total_episodes = 1_000_000  # Total episodes for training
max_steps = 1000  # Max possible steps in an episode
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 10_000  # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00005  # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95  # Discounting rate

# MEMORY HYPERPARAMETERS
# If you have GPU change to 1_million
pretrain_length = 10_000  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10_000  # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True
# MODIFY THIS TO TRUE IF YOU WANT TO SEE TRANING IN ACTION
training_render = False
# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PRE POPULATE MEMORY WITH RANDOM ACTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Reset the graph
tf.compat.v1.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = Agent.Agent(state_size, action_size, learning_rate, L, name="DQNetwork")

# Instantiate the target network
TargetNetwork = Agent.Agent(state_size, action_size, learning_rate, L, name="TargetNetwork")

# Saver will help us to save our model
saver = tf.compat.v1.train.Saver()

if training:

    print("training started...\n Pre populating memory.")

    # Instantiate memory
    memory = Agent.Memory(memory_size)

    # Render the environment
    Init()
    prev_world_score = 0
    done = False
    # First we need a state
    # state = np.zeros([*state_size])
    state = DQNetwork.observe(None, True)

    for i in range(pretrain_length):

        _stp = i + 1

        # Random action
        action = random.choice([0, 1, 2, 3])
        # print(action)
        # Make an action within the game
        r_locs, i_locs, c_locs, turret_ang, world_score = Game_step(action)
        observation = (r_locs, i_locs, c_locs, turret_ang, world_score, action)

        # Get the rewards
        # reward = calculate_reward(observation, prev_world_score)
        reward = DQNetwork.calculate_reward(observation)
        strikes = DQNetwork.calculate_strikes(world_score)

        # Look if the episode is finished
        if np.remainder(_stp, 1000) == 0:
            DQNetwork.strikes = 0
            # If we're dead
            done = True

            if world_score > 0:
                reward = 100.0
            elif strikes < 3 and world_score < 0:
                reward = -10.0
            elif strikes < 2 and world_score < 0:
                reward = -5.0
            else:
                reward = -100.0

            # We finished the episode
            # next_state = np.zeros([*state_size])
            next_state = DQNetwork.observe(None, True)

            # Add experience to memory
            experience = state, one_hot_actions[action], reward, next_state, done
            memory.store(experience)

            # Start a new episode
            Init()
            prev_world_score = 0
            done = False
            # First we need a state
            # state = np.zeros([*state_size])
            state = DQNetwork.observe(None, True)

        else:
            # we're not dead
            done = False

            # Get the next state
            # next_state = observe(observation)
            # next_state = DQNetwork.observe(observation)
            next_state = DQNetwork.observe(observation, False)

            # Add experience to memory
            experience = state, one_hot_actions[action], reward, next_state, done
            memory.store(experience)

            # Our state is now the next_state
            state = next_state

    print("pre-population finished.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAIN FOR NUMBER OF EPISODES USING DDQN Agent ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("initializing training simulations..")
    with tf.compat.v1.Session() as sess:
        # Initialize the variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Setup TensorBoard Writer
        writer = tf.compat.v1.summary.FileWriter('./tensorboard/dddqn/4', sess.graph)
        # Losses
        # for var in tf.trainable_variables():
        #     # print(var.eval())
        #     tf.compat.v1.summary.histogram(var.name, var.eval())

        tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)
        # tf.compat.v1.summary.scalar("IS-W", DQNetwork.ISWeights_)
        # tf.compat.v1.summary.scalar("Abs err", DQNetwork.absolute_errors)

        write_op = tf.compat.v1.summary.merge_all()

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Set tau = 0
        tau = 0

        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = Agent.update_target_graph()
        sess.run(update_target)

        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            # print("epizode: ", episode)
            # Initialize the rewards of t/he episode
            episode_rewards = []

            # Make a new episode and observe the first state
            Init()
            done = False
            prev_world_score = 0
            start = time.time()

            # state = np.zeros([*state_size])
            state = DQNetwork.observe(None, True)

            while step < max_steps and not done:
                step += 1

                # Increase the C step
                tau += 1

                # Increase decay_step
                decay_step += 1

                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability = Agent.Agent.predict_action(DQNetwork, explore_start,
                                                                         explore_stop, decay_rate,
                                                                         decay_step, state, sess)

                # image = tf.reshape(state, [-1, *state_size])
                # # # print(image.shape)
                # tf.compat.v1.summary.image('input', image, 3)

                # print(DQNetwork.Q.eval())
                # print(DQNetwork.output.eval())

                # Make an action within the game
                r_locs, i_locs, c_locs, turret_ang, world_score = Game_step(action)
                observation = (r_locs, i_locs, c_locs, turret_ang, world_score, action)

                if training_render:
                    Draw()

                # Get the rewards
                # reward = calculate_reward(observation, prev_world_score)
                reward = DQNetwork.calculate_reward(observation)
                strikes = DQNetwork.calculate_strikes(world_score)
                # print("strikes: ", strikes, "score: ", world_score)

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if step == max_steps or strikes == 3:
                    DQNetwork.strikes = 0
                    step = max_steps
                    done = True
                    # We finished the episode
                    # next_state = np.zeros([*state_size])
                    next_state = DQNetwork.observe(None, True)

                    if world_score > 0:
                        reward = 100.0
                    elif strikes < 3 and world_score < 0:
                        reward = -10.0
                    elif strikes < 2 and world_score < 0:
                        reward = -5.0
                    else:
                        reward = -100.0

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Score: {}'.format(world_score),
                          'Strikes: {}'.format(strikes),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Time: {:.4f}'.format((time.time() - start)))

                    # Add experience to memory
                    experience = state, one_hot_actions[action], reward, next_state, done
                    memory.store(experience)

                else:
                    # we're not finished
                    done = False

                    # Get the next state
                    # next_state = observe(observation)
                    # next_state = DQNetwork.observe(observation)
                    next_state = DQNetwork.observe(observation, False)

                    # Add experience to memory
                    # print(action)
                    experience = state, one_hot_actions[action], reward, next_state, done
                    memory.store(experience)

                    # st+1 is now our current state
                    state = next_state

                # LEARNING PART
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)

                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                # actions_mb.reshape([4, 64])
                # print(actions_mb)
                # print(rewards_mb)
                target_Qs_batch = []

                # DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')

                # Get Q values for next_state
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output,
                                               feed_dict={TargetNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])
                # print((DQNetwork.actions_: actions_mb))
                _, loss, absolute_errors = sess.run(
                    [DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                    feed_dict={DQNetwork.inputs_: states_mb,
                               DQNetwork.target_Q: targets_mb,
                               DQNetwork.actions_: actions_mb,
                               DQNetwork.ISWeights_: ISWeights_mb})

                # Update priority
                memory.batch_update(tree_idx, absolute_errors)
                # print(states_mb.shape)
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb,
                                                        DQNetwork.ISWeights_: ISWeights_mb,
                                                        TargetNetwork.inputs_: states_mb})

                # conv1_kernel_val = sess.graph.get_tensor_by_name('DQNetwork/conv1/kernel:0').eval()
                # # print(conv1_kernel_val)
                # tf.compat.v1.summary.scalar("kernele", conv1_kernel_val)

                # d2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DQNetwork.name)
                # # print(d2_vars[0])
                # tf.summary.histogram("weights", d2_vars[0])
                # tf.summary.histogram("biases", d2_vars[1])

                # gr = sess.graph
                # for op in gr.get_operations():
                #     print(op.name)

                # conv1_kernel_val = gr.get_tensor_by_name('DQNetwork/conv1/kernel:0').eval()
                # conv1_bias_val = gr.get_tensor_by_name('DQNetwork/conv1/bias:0').eval()
                # tf.summary.histogram("weights", conv1_kernel_val)
                # tf.summary.histogram("biases", conv1_bias_val)
                # for var in tf.trainable_variables():
                #     tf.summary.histogram(var.name, var)

                writer.add_summary(summary, episode)
                writer.flush()

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = Agent.update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, './models/model.ckpt')
                print("Model Saved")

    # writer.close()

else:

    with tf.compat.v1.Session() as sess:

        # Load the model
        saver.restore(sess, "./models/model.ckpt")

        writer = tf.compat.v1.summary.FileWriter('./tensorboard/dddqn/4', sess.graph)
        # Losses
        # for var in tf.trainable_variables():
        #     # print(var.eval())
        #     tf.compat.v1.summary.histogram(var.name, var.eval())

        # tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)
        # tf.compat.v1.summary.scalar("IS-W", DQNetwork.ISWeights_)
        # tf.compat.v1.summary.scalar("Abs err", DQNetwork.absolute_errors)

        write_op = tf.compat.v1.summary.merge_all()

        # play for 100 games
        for i in range(100):
            # Start the game
            Init()
            done = False
            score = 0
            prev_world_score = 0

            # state = np.zeros([*state_size])
            state = DQNetwork.observe(None, True)
            first = True
            step = 0

            while step < max_steps:
                step += 1

                # EPSILON GREEDY STRATEGY
                # Choose action a from state s using epsilon greedy.
                # First we randomize a number
                exp_exp_tradeoff = np.random.rand()

                explore_probability = 0.01

                if explore_probability > exp_exp_tradeoff:
                    # Make a random action (exploration)
                    action = random.choice([0, 1, 2, 3])

                else:
                    # Get action from Q-network (exploitation)
                    # Estimate the Qs values state
                    Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
                    action = np.argmax(Qs)

                # if step == 1:
                #     action = 3
                # else:
                #     action = 1
                # Make an action within the gameDXY
                # if action == 3: action = 1
                if first:
                    action = 3
                    first = False

                r_locs, i_locs, c_locs, turret_ang, world_score = Game_step(action)

                observation = (r_locs, i_locs, c_locs, turret_ang, world_score, action)

                if episode_render:
                    Draw()

                if step == max_steps:
                    score = world_score
                    break

                else:
                    # Get the next state
                    # next_state = observe(observation)
                    # next_state = DQNetwork.observe(observation)
                    next_state = DQNetwork.observe(observation, False)

                    reward = DQNetwork.calculate_reward(observation)
                    # print(reward)
                    # reward, prev_theta = calculate_reward(observation, prev_theta)
                    state = next_state

                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape)),
                                                        TargetNetwork.inputs_: state.reshape((1, *state.shape))})
                writer.add_summary(summary, i)
                writer.flush()

            print("Score: ", score)
