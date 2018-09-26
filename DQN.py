import gym
import numpy as np
import tensorflow as tf
import sys
import os
from lib import plotting

env = gym.envs.make("Breakout-v0")

# Atari Actions: 0 (noop), 1 (fire), 2 (left) and 3 (right) are valid actions
ACTIONS = [0, 1, 2, 3]
n_actions = len(ACTIONS)

class Image:
    """
    Process each Atari image. Resize it and convert it to grayscale
    """

    def __init__(self):
        with tf.variable_scope("image_processor"):
            self.input = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, image):
        processed = sess.run(self.output, {self.input: image})

        # Cropped image is [84, 84], but we need it [?, 84, 84, 4]
        processed = np.expand_dims(processed, axis=0)
        processed = np.stack([processed] * 4, axis=3)
        return processed


class Estimator:
    def __init__(self, scope='estimator', summaries_dir=None):
        self.scope = scope
        self.summaries_dir=summaries_dir

        with tf.variable_scope(scope):
            self.build_tf_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def build_tf_model(self):
        with tf.variable_scope("ConvNet"):
            # define placeholders for input
            self.X_pl = tf.placeholder(dtype=np.uint8, shape=[None, 84, 84, 4], name="X")
            self.y_pl = tf.placeholder(dtype=np.float32, shape=[None], name="y")
            self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

            batch_size = tf.shape(self.X_pl)[0]
            X = tf.to_float(self.X_pl) / 255.0

            # define conv layers
            conv1 = tf.layers.conv2d(X, 32, 8, 4, activation=tf.nn.relu, name="layer1")
            conv2 = tf.layers.conv2d(conv1, 64, 4, 2, activation=tf.nn.relu)
            conv3 = tf.layers.conv2d(conv2, 64, 3, 1, activation=tf.nn.relu)

            # fully connect layers, 512 neurones, then 4, because there are 4 actions
            fc = tf.layers.dense(tf.contrib.layers.flatten(conv3), 512)
            self.predictions = tf.layers.dense(fc, n_actions)

            gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
            self.actions_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

            # define loss function
            self.losses = tf.squared_difference(self.y_pl, self.actions_predictions)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

            # Summaries for Tensorboard
            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.histogram("q_values_hist", self.predictions),
                tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
            ])

    def predict(self, sess, s):
        return sess.run(self.predictions, {self.X_pl: s})

    def update(self, sess, s, a, y):
        a = np.expand_dims(a, axis=0)
        y = np.expand_dims(y, axis=0)

        feed_dict = {self.X_pl: s, self.y_pl: y, self.actions_pl: a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

        return loss


def make_epsilon_greedy_policy(sess, q_estimator, image_processor, epsilon, nA):
    def policy_fn(state):
        processed_state_image = image_processor.process(sess, state)
        q_values = q_estimator.predict(sess, processed_state_image)
        max_action = np.argmax(q_values)

        A = np.ones(nA, dtype=float) * epsilon / nA
        A[max_action] += (1.0 - epsilon)
        return A

    return policy_fn


def initialize_stats(num_episodes):
    return plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

def deep_q_learning(env, sess, q_estimator, image_processor,
                    num_episodes=100,
                    discount_factor=1.0,
                    epsilon=0.1,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    replay_memory_size=10000):

    # Initialize the stats we keep track of
    stats = initialize_stats(num_episodes)

    # Get the current time step
    current_timestep = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # Initialize the policy
    policy = make_epsilon_greedy_policy(sess, q_estimator, image_processor, epsilon, n_actions)

    replay_memory = []

    for i_episode in range(num_episodes):
        S = env.reset()

        t = 1
        done = False
        while not done:
            # Epsilon for this time step
            epsilon = epsilons[min(current_timestep, epsilon_decay_steps - 1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            q_estimator.summary_writer.add_summary(episode_summary, current_timestep)

            # Actual Q-Learning
            action_probs = policy(S)
            A = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            S_prime, reward, done, _ = env.step(A)
            processed_state_image = image_processor.process(sess, S_prime)
            target = reward + discount_factor * np.max(q_estimator.predict(sess, processed_state_image))
            loss = q_estimator.update(sess, processed_state_image, A, target)

            if (len(replay_memory) == replay_memory_size):
                replay_memory.pop(0)
            replay_memory.append(processed_state_image)

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                t, current_timestep, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            env.render()

            t += 1
            # Get into the next state
            S = S_prime

        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], node_name="episode_reward",
                                  tag="episode_reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], node_name="episode_length",
                                  tag="episode_length")
        q_estimator.summary_writer.add_summary(episode_summary, current_timestep)
        q_estimator.summary_writer.flush()

        yield current_timestep, plotting.EpisodeStats(
            episode_lengths=stats.episode_lengths[:i_episode + 1],
            episode_rewards=stats.episode_rewards[:i_episode + 1])


tf.reset_default_graph()

experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q", summaries_dir=experiment_dir)

image_processor = Image()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for t in deep_q_learning(env, sess, q_estimator, image_processor):
        print(1)
