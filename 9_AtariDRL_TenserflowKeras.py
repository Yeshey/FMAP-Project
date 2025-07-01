
# %%
# !pip install 'stable_baselines3'

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common import atari_wrappers
import ale_py

env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")

obs,_ = env.reset()

print(obs.shape)

env.render()

# %%
import numpy as np

# Gym dá no formato (H, W, C) que é o que o tensorflow espera

import typing as tt

class BufferWrapper(gym.ObservationWrapper): 
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        obs = env.observation_space
        assert isinstance(obs, spaces.Box)
        new_obs = gym.spaces.Box(
            #obs.low.repeat(n_steps, axis=0), obs.high.repeat(n_steps, axis=0),
            # antes os channels tavam no inicio agora estao no fim, agora temos (H, W, C * n_steps) não (C * n_steps, H, W)
            obs.low.repeat(n_steps, axis=-1), obs.high.repeat(n_steps, axis=-1),
            dtype=obs.dtype)

        self.observation_space = new_obs
        self.buffer = collections.deque(maxlen=n_steps)

    def reset(self, *, seed: tt.Optional[int] = None, options: tt.Optional[dict[str, tt.Any]] = None):
        for _ in range(self.buffer.maxlen-1): # preencher o buffer com frames vazias
            self.buffer.append(self.env.observation_space.low)
        obs, extra = self.env.reset() # reset gym env
        return self.observation(obs), extra

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.buffer.append(observation)
        return np.concatenate(list(self.buffer), axis=-1) # concat along channel (last in the list)
        #return np.concatenate(self.buffer)


def make_env(env_name: str, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = atari_wrappers.AtariWrapper(env, clip_reward=False, noop_max=0)
    #env = ImageToPyTorch(env)
    env = BufferWrapper(env, n_steps=4)
    return env

# %%
import tensorflow as tf
from tensorflow.keras import layers

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Is built with cuda:", tf.test.is_built_with_cuda())

class DQN(tf.keras.Model):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__() # ta a fazer init da superclass nn.Module

        print(f"DQN input shape: {input_shape}")

        self.rescale = layers.Rescaling(1./255) # normalização aqui
        self.conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=4, activation="relu", input_shape=input_shape)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=2, activation="relu")
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, activation="relu")
        self.flatten  = layers.Flatten()

        # fully connected
        self.fc = layers.Dense(units=512, activation='relu')
        self.out = layers.Dense(units=n_actions, activation = None)


    def call(self, inputs): # requiered for subclasses of tf.keras.Model
        x = self.rescale(inputs)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)

        x = self.fc(x)
        q = self.out(x)
        return q

# %%
from dataclasses import dataclass
from typing import Tuple

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0 # começamos com probabilidade 1 de fazer algo ao calhas
EPSILON_FINAL = 0.01 # acabamos com probabilidade 0.01 de fazer algo ao calhas

State = np.ndarray
Action = int
BatchTensors = tt.Tuple[
    tf.Tensor,           # current state (batch, H, W, C)
    tf.Tensor,           # actions
    tf.Tensor,               # rewards
    tf.Tensor,           # done || trunc
    tf.Tensor           # next state
]

@dataclass
class Experience:
    state: State
    action: Action
    reward: float
    done_trunc: bool
    new_state: State


class ExperienceBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> tt.List[Experience]:
        indices = np.random.choice(len(self), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

# %%
class Agent:
    def __init__(self, env: gym.Env, exp_buffer: ExperienceBuffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state: tt.Optional[np.ndarray] = None
        self._reset()

    def _reset(self):
        self.state, _ = env.reset()
        self.total_reward = 0.0

    # @torch.no_grad()
    def play_step(self, net: DQN,
                  epsilon: float = 0.0) -> tt.Optional[float]:
        done_reward = None

        if np.random.random() < epsilon: # com probabilidade epsilon joga ao calhas
            action = env.action_space.sample()
        else: # caso contrario usa a informação do Q

            #state_v = torch.as_tensor(self.state).to(device)
            state_v = tf.convert_to_tensor(self.state, dtype=tf.float32)

            #state_v.unsqueeze_(0) # gera uma dimensão, erro comum
            state_v = tf.expand_dims(state_v, axis=0)

            #q_vals_v = net(state_v) # gera os Qs
            q_values = net(state_v)  

            #_, act_v = torch.max(q_vals_v, dim=1) # queremos os maiores na dimensão 1
            act_v = tf.argmax(q_values, axis=1)

            #action = int(act_v.item()) # faz a ação que tem o melhor Q
            action = int(act_v[0])
            # action = int(act_idx.numpy()[0])

        # do step in the environment
        new_state, reward, is_done, is_tr, _ = self.env.step(action) # joga essa ação
        self.total_reward += reward

        exp = Experience( #
            state=self.state, action=action, reward=float(reward),
            done_trunc=is_done or is_tr, new_state=new_state
        )
        # informação esta guardada na forma (0,a,r,done,s'). e o buffer é uma lista dessas coisas
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done or is_tr:
            done_reward = self.total_reward
            self._reset()
        return done_reward

# %%
from typing import List

#def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
def batch_to_tensors(batch: List[Experience]) -> BatchTensors:
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    
    #states_t = torch.as_tensor(np.asarray(states))
    states_t = tf.convert_to_tensor(states, dtype=tf.float32)

    #actions_t = torch.LongTensor(actions)
    actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)

    #rewards_t = torch.FloatTensor(rewards)
    rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)

    #dones_t = torch.BoolTensor(dones)
    dones_t = tf.convert_to_tensor(dones, dtype=tf.bool)
    
    #new_states_t = torch.as_tensor(np.asarray(new_state))
    new_states_t = tf.convert_to_tensor(new_state, dtype=tf.float32)

    #return states_t.to(device), actions_t.to(device), rewards_t.to(device), \
    #       dones_t.to(device),  new_states_t.to(device)
    return states_t, actions_t, rewards_t, dones_t,  new_states_t

# %%
def calc_loss(batch: tt.List[Experience], net: DQN, tgt_net: DQN): # -> torch.Tensor:
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch)

    q_values = net(states_t)

    #indices = tf.stack([tf.range(BATCH_SIZE), actions_t], axis=1)
    # q_values has shape (B, n_actions)
    batch_range = tf.range(tf.shape(q_values)[0], dtype=actions_t.dtype)   # shape (B,)
    indices     = tf.stack([batch_range, actions_t], axis=1)               # shape (B,2)

    #state_action_values = tf.expand_dims(q_values, indices)
    state_action_values = tf.gather_nd(q_values, indices)

    next_q = tf.reduce_max(tgt_net(new_states_t), axis=1)
    next_q = next_q * tf.cast(tf.logical_not(dones_t), tf.float32)

    expected_state_action_values = next_q * GAMMA + rewards_t

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss = loss_fn(expected_state_action_values, state_action_values)

    return loss

# %%
import cv2
import argparse
import collections
import time
import numpy as np
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                    help="Name of the environment, default=" + DEFAULT_ENV_NAME)
args, _ = parser.parse_known_args()
# device = torch.device(args.dev)

#env = wrappers.make_env(args.env)
env = make_env(args.env)
print(f"Environment observation space: {env.observation_space.shape}")
net = DQN(env.observation_space.shape, env.action_space.n)
tgt_net = DQN(env.observation_space.shape, env.action_space.n)

# initialize both models weights by calling them with dummy input
dummy_input = tf.zeros((1,) + env.observation_space.shape, dtype=tf.float32)
print(f"Dummy input shape: {dummy_input.shape}")
net(dummy_input)  # This creates the weights
tgt_net(dummy_input)  # This creates the weights


log_dir = f"logs/{args.env}_{int(time.time())}"
writer = tf.summary.create_file_writer(log_dir)

#writer = SummaryWriter(comment="-" + args.env)
print("Network architecture:")
#net.build((None,) + env.observation_space.shape)
#net.summary()

buffer = ExperienceBuffer(REPLAY_SIZE)
agent = Agent(env, buffer)
epsilon = EPSILON_START

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
total_rewards = []
frame_idx = 0
ts_frame = 0
ts = time.time()
best_m_reward = None


while True:
    frame_idx += 1
    epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

    reward = agent.play_step(net, epsilon)
    if reward is not None:
        total_rewards.append(reward)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        m_reward = np.mean(total_rewards[-100:])
        print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, "
              f"eps {epsilon:.2f}, speed {speed:.2f} f/s")

        # TensorBoard logging for TensorFlow
        with writer.as_default():
            tf.summary.scalar("epsilon", epsilon, step=frame_idx)
            tf.summary.scalar("speed", speed, step=frame_idx)
            tf.summary.scalar("reward_100", m_reward, step=frame_idx)
            tf.summary.scalar("reward", reward, step=frame_idx)
            writer.flush()
        #writer.add_scalar("epsilon", epsilon, frame_idx)
        #writer.add_scalar("speed", speed, frame_idx)
        #writer.add_scalar("reward_100", m_reward, frame_idx)
        #writer.add_scalar("reward", reward, frame_idx)


        if best_m_reward is None or best_m_reward < m_reward:
            #torch.save(net.state_dict(), args.env + "-best_%.0f.dat" % m_reward)
            net.save_weights(args.env + "-best_%.0f.dat" % m_reward + ".weights.h5")
            if best_m_reward is not None:
                print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
            best_m_reward = m_reward
        if m_reward > MEAN_REWARD_BOUND:
            print("Solved in %d frames!" % frame_idx)
            break
    if len(buffer) < REPLAY_START_SIZE:
        continue

    # copy weights from net to tgt_net
    #if frame_idx % SYNC_TARGET_FRAMES == 0:
    #    tgt_net.load_state_dict(net.state_dict())
    if frame_idx % SYNC_TARGET_FRAMES == 0:
        tgt_net.set_weights(net.get_weights())

    # optimizer.zero_grad() # precisamos de fazer manualmente, se não acumulam, em tensorflow keras n temos
    batch = buffer.sample(BATCH_SIZE)

    #loss_t = calc_loss(batch, net, tgt_net)
    #loss_t.backward()
    #optimizer.step()
    with tf.GradientTape() as tape:
        loss_t = calc_loss(batch, net, tgt_net)
    grads = tape.gradient(loss_t, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))

    if frame_idx % 10 == 0:
        print(f"training in frame {frame_idx}, Loss = {loss_t:.4f}")
writer.close()
