import eventlet
eventlet.monkey_patch()

import math
import random
import threading
import time
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from flask import Flask, render_template
from flask_socketio import SocketIO

# --- Configuration ---
MAP_SIZE = 2000
FOOD_COUNT = 150
BOT_COUNT = 22
TICK_RATE = 30
MAX_BOT_MASS = 500
MIN_BOT_MASS = 15
STARTING_MASS = 25
SAVE_DIR = './saves/'
if os.path.exists('/content'):
    SAVE_DIR = '/content/drive/MyDrive/AlgorithmArena/saves/'
os.makedirs(SAVE_DIR, exist_ok=True)

# Shared Lock
game_state_lock = threading.Lock()

# --- Helpers ---
def safe_decide(bot, inputs):
    try:
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=0.0)
        action = bot.decide(inputs)
        if action is None or (isinstance(action, float) and np.isnan(action)):
            return 10 # STAY
        return int(np.clip(action, 0, 11))
    except Exception as e:
        print(f"⚠️ {bot.name} crashed during decision: {e}")
        return 10

class SpatialGrid:
    def __init__(self, map_size=2000, cell_size=200):
        self.cell_size = cell_size
        self.grid_dim = map_size // cell_size
        self.cells = [[[] for _ in range(self.grid_dim)] for _ in range(self.grid_dim)]
    
    def clear(self):
        for x in range(self.grid_dim):
            for y in range(self.grid_dim):
                self.cells[x][y] = []
    
    def get_cell(self, x, y):
        cx = min(self.grid_dim - 1, max(0, int(x // self.cell_size)))
        cy = min(self.grid_dim - 1, max(0, int(y // self.cell_size)))
        return (cx, cy)
    
    def add_entity(self, entity):
        cx, cy = self.get_cell(entity.x, entity.y)
        self.cells[cx][cy].append(entity)
    
    def get_nearby_entities(self, x, y):
        cx, cy = self.get_cell(x, y)
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.grid_dim and 0 <= ny < self.grid_dim:
                    nearby.extend(self.cells[nx][ny])
        return nearby

# Using simple classes instead of dataclass for older python compatibility if needed
class GameObject:
    def __init__(self, obj_id, x, y, mass=10, obj_type="generic", color="#FFFFFF"):
        self.id = obj_id
        self.x = x
        self.y = y
        self.mass = mass
        self.type = obj_type
        self.color = color

    @property
    def radius(self):
        return math.sqrt(max(1, self.mass)) * 2.5

class FoodPellet(GameObject):
    def __init__(self, fid, x, y):
        super().__init__(fid, x, y, mass=8.0, obj_type="food", color="#00FF00")

class Flag(GameObject):
    def __init__(self, fid, x, y, team_id):
        color = "#FF0000" if team_id == 1 else "#0000FF"
        super().__init__(fid, x, y, mass=40.0, obj_type="flag", color=color)
        self.team_id = team_id
        self.carrier = None

class Checkpoint(GameObject):
    def __init__(self, fid, x, y, sequence_num):
        super().__init__(fid, x, y, mass=30.0, obj_type="checkpoint", color="#FFFF00")
        self.sequence_num = sequence_num

class Resource(GameObject):
    def __init__(self, fid, x, y, value, color):
        super().__init__(fid, x, y, mass=15.0, obj_type="resource", color=color)
        self.value = value

class PuzzleElement(GameObject):
    def __init__(self, fid, x, y, element_type):
        super().__init__(fid, x, y, mass=25.0, obj_type="puzzle", color="#A0A0A0")
        self.element_type = element_type # "button", "plate"
        self.activated = False

class GameMode:
    def __init__(self, world):
        self.world = world
        self.name = "Abstract Mode"
        self.episode_length = 3000 # Default frames
        self.features = {} # Mode specific recognition features

    def initialize(self):
        """Called at the start of an episode to setup the arena."""
        pass

    def update(self):
        """Called every frame to update mode-specific logic."""
        pass

    def calculate_reward(self, bot):
        """Returns the reward for a specific bot in this frame."""
        return 0

    def check_victory(self):
        """Returns victory info if the game should end."""
        if self.world.frame >= self.episode_length:
            return {"reason": "time_limit"}
        return None

    def get_state_for_ui(self):
        """Returns mode-specific data for the frontend."""
        return {}

class BaseBot:
    def __init__(self, bot_id):
        self.id = bot_id
        self.name = f"Bot-{bot_id}"
        self.algo_name = "Base"
        self.color = "#FFFFFF"
        self.x = random.uniform(100, MAP_SIZE - 100)
        self.y = random.uniform(100, MAP_SIZE - 100)
        self.mass = STARTING_MASS
        self.velocity = [0.0, 0.0]
        self.food_eaten = 0
        self.kills = 0
        self.deaths = 0
        self.time_alive_current = 0
        self.max_mass_achieved = STARTING_MASS
        self.last_reward = 0
        self.total_reward = 0
        self.vision_rays = 24
        self.vision_range = 500
        self.ray_results = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(24)]
        # Meta-learning additions
        self.team_id = 0 # 0: neutral, 1: red, 2: blue, 3: zombie, etc.
        self.role = "none" # Mode-specific role
        self.status = {} # Arbitrary status flags
        self.died = False

    @property
    def radius(self):
        return math.sqrt(max(1, self.mass)) * 2.5

    def update_physics(self):
        self.time_alive_current += 1
        if self.mass > self.max_mass_achieved:
            self.max_mass_achieved = self.mass
        speed_mod = 5.0 / (1.0 + self.mass * 0.005)
        self.x += self.velocity[0] * speed_mod
        self.y += self.velocity[1] * speed_mod

    def receive_reward(self, reward):
        self.last_reward += reward
        self.total_reward += reward

    def sense(self, world):
        self.ray_results = []
        for i in range(self.vision_rays):
            angle = math.radians(i * (360 / self.vision_rays))
            dx, dy = math.cos(angle), math.sin(angle)

            # [dist, wall, food, enemy, team, special, value, size, rel_mass]
            res = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Wall
            wall_dist = float('inf')
            if dx > 0: wall_dist = min(wall_dist, (world.width - self.x) / dx)
            elif dx < 0: wall_dist = min(wall_dist, -self.x / dx)
            if dy > 0: wall_dist = min(wall_dist, (world.height - self.y) / dy)
            elif dy < 0: wall_dist = min(wall_dist, -self.y / dy)
            if wall_dist < self.vision_range:
                res[0] = wall_dist / self.vision_range
                res[1] = 1.0

            # Objects
            nearby = world.grid.get_nearby_entities(self.x, self.y)
            for other in nearby:
                if other == self: continue
                dist = math.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)
                if dist >= self.vision_range: continue

                target_angle = math.atan2(other.y - self.y, other.x - self.x)
                diff = abs((target_angle - angle + math.pi) % (2*math.pi) - math.pi)

                if diff < math.radians(7.5):
                    norm_dist = dist / self.vision_range
                    if norm_dist < res[0]:
                        res = [norm_dist, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        if isinstance(other, FoodPellet):
                            res[2] = 1.0
                        elif isinstance(other, BaseBot):
                            if self.team_id != 0 and self.team_id == other.team_id: res[4] = 1.0
                            else: res[3] = 1.0
                            res[7] = other.mass / 500.0
                            res[8] = other.mass / max(1, self.mass)
                        elif isinstance(other, (Flag, Checkpoint, Resource, PuzzleElement)):
                            res[5] = 1.0
                            if isinstance(other, Flag): res[6] = other.team_id / 5.0
                            elif isinstance(other, Checkpoint):
                                if self.status.get("next_checkpoint") == other.sequence_num: res[6] = 1.0
                                else: res[6] = 0.5
                            elif isinstance(other, Resource): res[6] = other.value / 5.0
                            elif isinstance(other, PuzzleElement): res[6] = 1.0 if other.activated else 0.0
                            res[7] = other.mass / 500.0
            self.ray_results.append(res)

    def get_input_vector(self):
        inputs = []
        for r in self.ray_results: inputs.extend(r)
        inputs.extend([
            self.mass / 500.0, self.velocity[0], self.velocity[1],
            self.x / MAP_SIZE, self.y / MAP_SIZE, self.team_id / 5.0,
            1.0 if self.last_reward > 0 else (-1.0 if self.last_reward < 0 else 0.0),
            1.0 if self.died else 0.0, 0.0, 1.0
        ])
        return np.array(inputs, dtype=np.float32)

    def apply_action(self, action):
        if action < 8:
            a = math.radians(action * 45)
            self.velocity = [math.cos(a), math.sin(a)]
        elif action == 8: self.velocity = [v * 1.8 for v in self.velocity]; self.mass -= 0.02
        elif action == 9: self.velocity = [v * 0.5 for v in self.velocity]
        elif action == 10: self.velocity = [v * 0.9 for v in self.velocity]
        elif action == 11:
            a = random.uniform(0, 2*math.pi)
            self.velocity = [math.cos(a), math.sin(a)]

    def decide(self, inputs): return 10
    def get_internal_metric(self): return ""
    def save_state(self): return {}
    def load_state(self, state): pass

# --- Implementations (Classical) ---

class RandomBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Random"; self.color = "#888888"
        self.change_int = random.randint(15, 60); self.ticks = 0; self.act = random.randint(0, 11)
    def decide(self, inputs):
        self.ticks += 1
        if self.ticks >= self.change_int: self.act = random.randint(0, 11); self.ticks = 0
        return self.act

class RuleBasedBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Rules"; self.color = "#8B4513"; self.st = "EXP"
    def decide(self, inputs):
        for i in range(24):
            d = inputs[i*9]
            if inputs[i*9+3] == 1.0 and d < 0.2: self.st = "FLEE"; return (i * 15 + 180) // 45 % 8
            if inputs[i*9+2] == 1.0 and d < 0.5: self.st = "FEED"; return (i * 15) // 45 % 8
        self.st = "EXP"; return 2

class PotentialFieldBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Field"; self.color = "#00FFFF"
    def decide(self, inputs):
        fx, fy = 0.0, 0.0
        for i in range(24):
            d = inputs[i*9]; a = math.radians(i*15); dx, dy = math.cos(a), math.sin(a)
            dr = d * 500 + 1
            if inputs[i*9+2] == 1.0: fx += dx*50/(dr**1.5); fy += dy*50/(dr**1.5)
            elif inputs[i*9+3] == 1.0: fx -= dx*500/(dr**2); fy -= dy*500/(dr**2)
            elif inputs[i*9+1] == 1.0: fx -= dx*200/(dr**2); fy -= dy*200/(dr**2)
        if abs(fx) + abs(fy) < 0.01: return 10
        return int((math.degrees(math.atan2(fy, fx)) + 360) % 360 // 45) % 8

class PIDControllerBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "PID"; self.color = "#FFA500"
        self.integ, self.last_e = 0, 0
    def decide(self, inputs):
        ta = 0; min_d = 1.0
        for i in range(24):
            if inputs[i*9+2] == 1.0 and inputs[i*9] < min_d: min_d = inputs[i*9]; ta = i*15
        cur = math.degrees(math.atan2(self.velocity[1], self.velocity[0]))
        err = (ta - cur + 180) % 360 - 180
        self.integ += err; der = err - self.last_e; self.last_e = err
        out = 0.8 * err + 0.02 * self.integ + 0.3 * der
        return int((cur + out + 360) % 360 // 45) % 8

class GeneticBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Genetic"; self.color = "#00FF00"
        self.w1 = np.random.randn(226, 64) * 0.1; self.w2 = np.random.randn(64, 12) * 0.1; self.gen = 0
    def decide(self, inputs):
        x = np.maximum(0, inputs @ self.w1)
        l = x @ self.w2; e = np.exp(l - np.max(l))
        return np.random.choice(12, p=e/np.sum(e))
    def receive_reward(self, r):
        super().receive_reward(r)
        if r > 10 and random.random() < 0.2: self._mut(0.05)
        if r < -100: self._mut(0.2)
    def _mut(self, s): self.w1 += np.random.randn(*self.w1.shape)*s; self.w2 += np.random.randn(*self.w2.shape)*s; self.gen += 1
    def get_internal_metric(self): return f"G: {self.gen}"
    def save_state(self): return {'w1': self.w1.tolist(), 'w2': self.w2.tolist(), 'gen': self.gen}
    def load_state(self, s): self.w1 = np.array(s['w1']); self.w2 = np.array(s['w2']); self.gen = s['gen']

class TabularQLearningBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Q-Table"; self.color = "#FFFF00"
        self.q = {}; self.eps = 0.2; self.ls, self.la = None, None
    def _disc(self, i):
        mf, fa, mt, ta = 1.0, 0, 1.0, 0
        for r in range(24):
            if i[r*9+2] == 1.0 and i[r*9] < mf: mf, fa = i[r*9], r
            if i[r*9+3] == 1.0 and i[r*9] < mt: mt, ta = i[r*9], r
        return (fa//3, int(mf*4), ta//3, int(mt*4))
    def decide(self, i):
        s = self._disc(i); self.ls = s
        if random.random() < self.eps: self.la = random.randint(0, 11)
        else: self.la = np.argmax(self.q.get(s, np.zeros(12)))
        return self.la
    def learn(self, r, ni):
        if self.ls is None: return
        ns = self._disc(ni); oq = self.q.get(self.ls, np.zeros(12)); nq = self.q.get(ns, np.zeros(12))
        oq[self.la] += 0.1 * (r + 0.9 * np.max(nq) - oq[self.la]); self.q[self.ls] = oq
    def get_internal_metric(self): return f"Q: {len(self.q)}"
    def save_state(self): return {'q': {str(k): v.tolist() for k,v in self.q.items()}, 'eps': self.eps}
    def load_state(self, s): self.q = {eval(k): np.array(v) for k,v in s['q'].items()}; self.eps = s['eps']

# --- Implementations (Deep RL) ---

class DQNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(226, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 12))
    def forward(self, x): return self.net(x)

class DeepQBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "DQN"; self.color = "#FF0000"
        self.model = DQNet(); self.target = DQNet(); self.target.load_state_dict(self.model.state_dict())
        self.opt = optim.Adam(self.model.parameters(), lr=0.0005); self.mem = deque(maxlen=50000)
        self.eps, self.steps, self.ls, self.la = 1.0, 0, None, None
    def decide(self, i):
        self.steps += 1; self.ls = i
        if random.random() < self.eps: self.la = random.randint(0, 11)
        else: 
            with torch.no_grad(): self.la = torch.argmax(self.model(torch.FloatTensor(i))).item()
        return self.la
    def learn(self, r, ni, d):
        if self.ls is None: return
        self.mem.append((self.ls, self.la, r, ni, d))
        if len(self.mem) > 128 and self.steps % 4 == 0:
            b = random.sample(self.mem, 128); s, a, r, ns, d = zip(*b)
            s, a, r, ns, d = torch.FloatTensor(np.array(s)), torch.LongTensor(a), torch.FloatTensor(r), torch.FloatTensor(np.array(ns)), torch.BoolTensor(d)
            cq = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
            nq = self.target(ns).max(1)[0]; nq[d] = 0; tq = r + 0.99 * nq
            loss = nn.MSELoss()(cq, tq.detach()); self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.eps = max(0.05, self.eps * 0.9999)
            if self.steps % 500 == 0: self.target.load_state_dict(self.model.state_dict())
    def get_internal_metric(self): return f"E: {self.eps:.2f}"
    def save_state(self): return {'model': self.model.state_dict(), 'eps': self.eps}
    def load_state(self, s): self.model.load_state_dict(s['model']); self.eps = s['eps']

class ACNet(nn.Module):
    def __init__(self):
        super().__init__(); self.base = nn.Sequential(nn.Linear(226, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): b = self.base(x); return self.actor(b), self.critic(b)

class ActorCriticBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "A2C"; self.color = "#8B00FF"
        self.model = ACNet(); self.opt = optim.Adam(self.model.parameters(), lr=0.001)
        self.llp, self.lv = None, None
    def decide(self, i):
        p, v = self.model(torch.FloatTensor(i)); d = torch.distributions.Categorical(p); a = d.sample()
        self.llp, self.lv = d.log_prob(a), v; return a.item()
    def learn(self, r, ni, d):
        if self.llp is None: return
        _, nv = self.model(torch.FloatTensor(ni)); t = r + (0 if d else 0.99 * nv.item())
        adv = t - self.lv.item()
        loss = -self.llp * adv + 0.5 * nn.MSELoss()(self.lv.squeeze(), torch.FloatTensor([t]))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
    def save_state(self): return {'model': self.model.state_dict()}
    def load_state(self, s): self.model.load_state_dict(s['model'])

# --- Search & Ensemble ---

class HeuristicSearchBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Search"; self.color = "#0000FF"; self.sc = 0
    def decide(self, i):
        best, ms = 10, -999
        for a in [0, 2, 4, 6, 8, 10]:
            s = 0
            for r in range(24):
                d = i[r*9]; ra = r*15
                if i[r*9+2] == 1.0 and abs(ra - a*45) < 45: s += 10/(d+0.1)
                if i[r*9+3] == 1.0 and abs(ra - a*45) < 45: s -= 50/(d+0.1)
            if s > ms: ms, best = s, a
        self.sc = ms; return best

class EnsembleBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Ensemble"; self.color = "#FF69B4"
    def decide(self, i):
        v1 = random.randint(0, 7); mf, fa = 1.0, 0
        for r in range(24):
            if i[r*9+2] == 1.0 and i[r*9] < mf: mf, fa = i[r*9], r
        v2 = fa//3; mt, ta = 1.0, 0
        for r in range(24):
            if i[r*9+3] == 1.0 and i[r*9] < mt: mt, ta = i[r*9], r
        v3 = (ta//3 + 4) % 8 if mt < 0.3 else v2
        c = {}; 
        for v in [v1, v2, v3]: c[v] = c.get(v, 0) + 1
        return max(c, key=c.get)

# --- META-LEARNING ARCHITECTURE ---

class SharedEncoder(nn.Module):
    def __init__(self, input_dim=226, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class CRN(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=64, num_modes=10):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_modes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return torch.softmax(self.fc(out[:, -1, :]), dim=-1)

class MSPN(nn.Module):
    def __init__(self, latent_dim=128, action_dim=12):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.actor(x), self.critic(x)

class MCN(nn.Module):
    def __init__(self, num_modes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(num_modes + 5, 32), nn.ReLU(), nn.Linear(32, num_modes), nn.Softmax(dim=-1))
    def forward(self, mode_probs, meta_stats):
        return self.net(torch.cat([mode_probs, meta_stats], dim=-1))

class MetaBot(BaseBot):
    def __init__(self, bot_id, encoder, crn, mspns, mcn):
        super().__init__(bot_id)
        self.algo_name = "Meta-Learner"
        self.encoder = encoder
        self.crn = crn
        self.mspns = mspns # ModuleList of 10 MSPNs
        self.mcn = mcn
        self.obs_buffer = deque(maxlen=100)
        self.mode_probs = torch.zeros(10)
        self.active_mode_idx = 0
        self.confidence = 0.0
        self.color = "#FFFFFF"

    def decide(self, inputs):
        self.obs_buffer.append(inputs)
        device = next(self.encoder.parameters()).device
        st = torch.FloatTensor(inputs).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = self.encoder(st)

            # Recognition (Throttle: only run every 20 steps to save CPU)
            if len(self.obs_buffer) >= 20 and len(self.obs_buffer) % 20 == 0:
                seq = torch.FloatTensor(np.array(list(self.obs_buffer))).unsqueeze(0).to(device)
                latent_seq = self.encoder(seq)
                self.mode_probs = self.crn(latent_seq).squeeze()
                self.active_mode_idx = torch.argmax(self.mode_probs).item()
                self.confidence = self.mode_probs[self.active_mode_idx].item()

            # In a real implementation, MCN would blend policies here
            # For now, we use the expert head of the recognized mode
            probs, _ = self.mspns[self.active_mode_idx](latent)
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

    def get_internal_metric(self):
        modes = ["Classic", "Tag", "TDM", "CTF", "KotH", "BR", "Inf", "Res", "Race", "Puz"]
        if self.confidence < 0.3: return "Scanning..."
        return f"{modes[self.active_mode_idx]} ({self.confidence*100:.0f}%)"

# --- NOVEL BOT (Episodic Synergy Reinforcement) ---

class SynergyNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(226+12, 128), nn.ReLU(), nn.Linear(128, 12))
    def forward(self, s, a_oh): return self.net(torch.cat([s, a_oh], dim=-1))

class NovelBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Novel"; self.color = "#FFD700"
        self.syn = SynergyNet(); self.opt = optim.Adam(self.syn.parameters(), lr=0.001)
        self.eps_mem = []; self.cur_traj = []; self.re_mode = False; self.re_idx = 0; self.active_ep = None
    def decide(self, i):
        st = torch.FloatTensor(i)
        if not self.re_mode and self.eps_mem:
            for ep in self.eps_mem:
                if np.linalg.norm(i - ep['s']) < 0.4: self.re_mode, self.active_ep, self.re_idx = True, ep, 0; break
        if self.re_mode:
            ba = self.active_ep['acts'][self.re_idx]; ba_oh = torch.zeros(12); ba_oh[ba] = 1.0
            with torch.no_grad(): off = self.syn(st, ba_oh)
            l = torch.zeros(12); l[ba] = 3.0; l += off
            a = torch.argmax(l).item(); self.re_idx += 1
            if self.re_idx >= len(self.active_ep['acts']): self.re_mode = False
        else: a = random.randint(0, 11)
        self.ls, self.la = i, a; return a
    def learn(self, r, ni, d):
        self.cur_traj.append((self.ls, self.la, r))
        if d or len(self.cur_traj) >= 50:
            tr = sum(x[2] for x in self.cur_traj)
            if tr > 40:
                self.eps_mem.append({'s': self.cur_traj[0][0], 'acts': [x[1] for x in self.cur_traj], 'r': tr})
                self.eps_mem.sort(key=lambda x: x['r'], reverse=True); self.eps_mem = self.eps_mem[:30]
            # Actual Synergy Training
            if len(self.cur_traj) > 5:
                # Simple Policy Gradient style update for Synergy Net
                s, a, r_seq = zip(*self.cur_traj); s = torch.FloatTensor(np.array(s))
                a_oh = torch.zeros(len(a), 12); 
                for idx, act in enumerate(a): a_oh[idx, act] = 1.0
                out = self.syn(s, a_oh); 
                # Loss = -log_prob * reward. Here we simplify:
                loss = (out.pow(2)).mean() * 0.01 # L2 regularizer
                # If reward > threshold, reinforce
                if tr > 50: loss -= (out.sum()) * 0.001 
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.cur_traj = []
    def get_internal_metric(self): return f"EP: {len(self.eps_mem)}"
    def save_state(self): return {'syn': self.syn.state_dict(), 'eps': [{'s': e['s'].tolist(), 'acts': e['acts'], 'r': e['r']} for e in self.eps_mem]}
    def load_state(self, s): 
        self.syn.load_state_dict(s['syn'])
        self.eps_mem = [{'s': np.array(e['s']), 'acts': e['acts'], 'r': e['r']} for e in s['eps']]

# --- TRAINING INFRASTRUCTURE ---

class HeadlessArena:
    def __init__(self, mode_class, bots):
        self.world = GameWorld()
        self.world.bots = bots
        self.world.set_mode(mode_class)

    def step(self):
        # Sense, Decide, Apply, Update
        experiences = []
        for b in self.world.bots:
            if b.died: continue
            old_state = b.get_input_vector()
            b.sense(self.world)
            action = safe_decide(b, old_state)
            b.apply_action(action)
            experiences.append((old_state, action, b))

        self.world.update()

        results = []
        for old_state, action, b in experiences:
            new_state = b.get_input_vector()
            reward = b.last_reward
            done = b.died
            results.append((old_state, action, reward, new_state, done))
            b.last_reward = 0

        return results, self.world.current_mode.check_victory()

def worker_process(worker_id, experience_queue, weights_dict, cmd_queue):
    # This is a simplified worker for multiprocessing training
    # In a real scenario, we'd use torch.multiprocessing and shared memory
    print(f"👷 Worker {worker_id} started.")
    # Local initialization...
    pass

# --- World & Server ---

class GameWorld:
    def __init__(self):
        self.width, self.height = MAP_SIZE, MAP_SIZE
        self.bots = []
        self.objects = []
        self.food = [] # We'll keep food separate for compatibility with existing sense() for now
        self.grid = SpatialGrid()
        self.frame = 0
        self.current_mode = None

    def set_mode(self, mode_class):
        self.current_mode = mode_class(self)
        self.objects = []
        self.food = []
        self.frame = 0
        self.current_mode.initialize()

    def spawn_food(self):
        f = FoodPellet(random.randint(0, 1000000), random.uniform(20, self.width-20), random.uniform(20, self.height-20))
        self.food.append(f)

    def update(self):
        self.frame += 1
        self.grid.clear()
        for b in self.bots:
            if not b.died: self.grid.add_entity(b)
        for obj in self.objects: self.grid.add_entity(obj)
        for f in self.food: self.grid.add_entity(f)

        # Mode update
        if self.current_mode:
            self.current_mode.update()

        for b in self.bots:
            if b.died: continue
            b.update_physics()
            # Wall Collision
            m = b.radius
            if b.x < m: b.x, b.velocity[0] = m, 0; b.receive_reward(-3)
            elif b.x > self.width-m: b.x, b.velocity[0] = self.width-m, 0; b.receive_reward(-3)
            if b.y < m: b.y, b.velocity[1] = m, 0; b.receive_reward(-3)
            elif b.y > self.height-m: b.y, b.velocity[1] = self.height-m, 0; b.receive_reward(-3)

            # Game Mode Reward
            if self.current_mode:
                b.receive_reward(self.current_mode.calculate_reward(b))

            # Combat logic
            for other in self.grid.get_nearby_entities(b.x, b.y):
                if b == other: continue
                dist = math.sqrt((b.x-other.x)**2 + (b.y-other.y)**2)
                if dist < b.radius:
                    if isinstance(other, BaseBot):
                        # Team check
                        if b.team_id != 0 and b.team_id == other.team_id: continue

                        if b.mass > other.mass * 1.25:
                            b.mass = min(500, b.mass + other.mass * 0.8)
                            b.kills += 1; b.receive_reward(150)
                            self.respawn(other)
                    elif isinstance(other, FoodPellet):
                        b.mass = min(500, b.mass + 8)
                        b.food_eaten += 1; b.receive_reward(15)
                        if other in self.food:
                            self.food.remove(other)
                            self.spawn_food()

            # Passive Mass Decay
            b.mass = max(15, b.mass - b.mass * 0.0005)

    def respawn(self, b):
        b.receive_reward(-200); b.deaths += 1
        # Drop flags if carrying
        for obj in self.objects:
            if isinstance(obj, Flag) and obj.carrier == b:
                obj.carrier = None

        if self.current_mode and getattr(self.current_mode, "allow_respawn", True) == False:
            b.died = True
            b.mass = 0
            return

        b.x, b.y = random.uniform(100, self.width-100), random.uniform(100, self.height-100)
        b.mass, b.velocity = 25, [0.0, 0.0]

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
world = GameWorld()

@app.route('/')
def index(): return render_template('index.html')

def game_loop():
    print("Loop started.")
    while True:
        st = time.time()
        with game_state_lock:
            for b in world.bots:
                b.sense(world)
                a = safe_decide(b, b.get_input_vector())
                b.apply_action(a)
            world.update()
            for b in world.bots:
                if hasattr(b, 'learn'):
                    if isinstance(b, TabularQLearningBot):
                        b.learn(b.last_reward, b.get_input_vector())
                    else:
                        b.learn(b.last_reward, b.get_input_vector(), False)
                b.last_reward = 0
            if world.frame % 2 == 0:
                bot_data = []
                for b in world.bots:
                    if b.died: continue
                    d = {
                        'name': b.name, 'algo_name': b.algo_name, 'color': b.color,
                        'x': b.x, 'y': b.y, 'mass': b.mass, 'radius': b.radius,
                        'metric': b.get_internal_metric(), 'team_id': b.team_id
                    }
                    if hasattr(b, 'ray_results'):
                        d['rays'] = b.ray_results # Send vision rays for attention viz
                    bot_data.append(d)

                socketio.emit('game_state', {
                    'frame': world.frame,
                    'mode': world.current_mode.name if world.current_mode else "None",
                    'bots': bot_data,
                    'food': [{'x': f.x, 'y': f.y} for f in world.food],
                    'objects': [{'type': o.type, 'x': o.x, 'y': o.y, 'radius': o.radius, 'color': o.color, 'activated': getattr(o, 'activated', False)} for o in world.objects]
                })
        if world.frame % 1000 == 0:
            for b in world.bots:
                st_data = b.save_state()
                if st_data: torch.save(st_data, f"{SAVE_DIR}{b.name}.pth")
            import psutil
            mem = psutil.virtual_memory().percent
            if mem > 80: print(f"⚠️ HIGH MEMORY: {mem}%")
        time.sleep(max(0, 1/30 - (time.time() - st)))

class ClassicArena(GameMode):
    def initialize(self):
        self.name = "Classic Arena"
        self.world.objects = []
        for _ in range(FOOD_COUNT): self.world.spawn_food()

class TagMode(GameMode):
    def initialize(self):
        self.name = "Tag"
        self.world.objects = []
        self.world.food = []
        if not self.world.bots: return
        self.it_bot = random.choice(self.world.bots)
        for b in self.world.bots:
            if b == self.it_bot:
                b.team_id = 3 # Special "IT" team
                b.color = "#FF4500"
            else:
                b.team_id = 0
                b.color = "#FFFFFF"

    def update(self):
        it = None
        for b in self.world.bots:
            if b.team_id == 3: it = b; break
        if not it: return

        for b in self.world.bots:
            if b == it: continue
            dist = math.sqrt((it.x - b.x)**2 + (it.y - b.y)**2)
            if dist < (it.radius + b.radius):
                it.receive_reward(100)
                it.team_id = 0; it.color = "#FFFFFF"
                b.receive_reward(-50)
                b.team_id = 3; b.color = "#FF4500"
                break

    def calculate_reward(self, bot):
        if bot.team_id == 3: return -0.1
        return 0.1

class TeamDeathmatchMode(GameMode):
    def initialize(self):
        self.name = "Team Deathmatch"
        self.world.objects = []
        self.world.food = []
        for i, b in enumerate(self.world.bots):
            if i < len(self.world.bots) // 2:
                b.team_id = 1; b.color = "#FF0000"
            else:
                b.team_id = 2; b.color = "#0000FF"

    def calculate_reward(self, bot):
        # Additional reward for proximity to teammates
        reward = 0
        for other in self.world.bots:
            if other != bot and other.team_id == bot.team_id:
                dist = math.sqrt((bot.x - other.x)**2 + (bot.y - other.y)**2)
                if dist < 200: reward += 0.01
        return reward

class CaptureTheFlagMode(GameMode):
    def initialize(self):
        self.name = "Capture the Flag"
        self.world.food = []
        self.world.objects = [
            Flag(1, 150, 150, 1),
            Flag(2, MAP_SIZE - 150, MAP_SIZE - 150, 2)
        ]
        for i, b in enumerate(self.world.bots):
            if i < len(self.world.bots) // 2:
                b.team_id = 1; b.color = "#FF0000"
            else:
                b.team_id = 2; b.color = "#0000FF"

    def update(self):
        for obj in self.world.objects:
            if isinstance(obj, Flag):
                if obj.carrier:
                    obj.x, obj.y = obj.carrier.x, obj.carrier.y
                    base_x, base_y = (150, 150) if obj.carrier.team_id == 1 else (MAP_SIZE-150, MAP_SIZE-150)
                    if math.sqrt((obj.x-base_x)**2 + (obj.y-base_y)**2) < 100:
                        obj.carrier.receive_reward(500)
                        obj.carrier = None
                        obj.x, obj.y = (150, 150) if obj.team_id == 1 else (MAP_SIZE-150, MAP_SIZE-150)
                else:
                    for b in self.world.bots:
                        if b.team_id != obj.team_id:
                            if math.sqrt((b.x-obj.x)**2 + (b.y-obj.y)**2) < b.radius + 20:
                                obj.carrier = b
                                b.receive_reward(100)
                                break

    def calculate_reward(self, bot):
        for obj in self.world.objects:
            if isinstance(obj, Flag) and obj.carrier == bot: return 0.5
        return 0

class KingOfTheHillMode(GameMode):
    def initialize(self):
        self.name = "King of the Hill"
        self.world.objects = []
        self.world.food = []
        self.hill_x, self.hill_y, self.hill_r = MAP_SIZE//2, MAP_SIZE//2, 250

    def calculate_reward(self, bot):
        dist = math.sqrt((bot.x - self.hill_x)**2 + (bot.y - self.hill_y)**2)
        if dist < self.hill_r: return 1.0
        return -0.1

class BattleRoyaleMode(GameMode):
    def initialize(self):
        self.name = "Battle Royale"
        self.allow_respawn = False
        self.zone_radius = MAP_SIZE * 0.8
        self.world.objects = []
        for _ in range(FOOD_COUNT // 2): self.world.spawn_food()

    def update(self):
        self.zone_radius = max(100, self.zone_radius - 0.5)
        center = MAP_SIZE // 2
        for b in self.world.bots:
            if b.died: continue
            dist = math.sqrt((b.x - center)**2 + (b.y - center)**2)
            if dist > self.zone_radius:
                b.mass -= 0.5
                b.receive_reward(-1)
                if b.mass <= MIN_BOT_MASS:
                    self.world.respawn(b)

    def calculate_reward(self, bot):
        if not bot.died: return 0.2
        return 0

class InfectionMode(GameMode):
    def initialize(self):
        self.name = "Infection"
        self.world.objects = []
        self.world.food = []
        if not self.world.bots: return
        zombie = random.choice(self.world.bots)
        for b in self.world.bots:
            if b == zombie:
                b.team_id = 3 # Zombie
                b.color = "#00FF00"
            else:
                b.team_id = 1 # Human
                b.color = "#FFFFFF"

    def update(self):
        for b in self.world.bots:
            if b.team_id == 3:
                for other in self.world.bots:
                    if other.team_id == 1:
                        dist = math.sqrt((b.x - other.x)**2 + (b.y - other.y)**2)
                        if dist < (b.radius + other.radius):
                            other.team_id = 3
                            other.color = "#00FF00"
                            b.receive_reward(100)
                            other.receive_reward(-50)

    def calculate_reward(self, bot):
        if bot.team_id == 3: return 0.1
        return 0.2

class ResourceCollectorMode(GameMode):
    def initialize(self):
        self.name = "Resource Collector"
        self.world.objects = []
        self.world.food = []
        for _ in range(30):
            val = random.choice([1, 3, 5])
            col = "#CD7F32" if val == 1 else ("#C0C0C0" if val == 3 else "#FFD700")
            self.world.objects.append(Resource(random.randint(0, 1000000), random.uniform(50, MAP_SIZE-50), random.uniform(50, MAP_SIZE-50), val, col))

    def update(self):
        for b in self.world.bots:
            for obj in self.world.objects[:]:
                if isinstance(obj, Resource):
                    dist = math.sqrt((b.x - obj.x)**2 + (b.y - obj.y)**2)
                    if dist < (b.radius + obj.radius):
                        b.receive_reward(obj.value * 20)
                        self.world.objects.remove(obj)
                        val = random.choice([1, 3, 5])
                        col = "#CD7F32" if val == 1 else ("#C0C0C0" if val == 3 else "#FFD700")
                        self.world.objects.append(Resource(random.randint(0, 1000000), random.uniform(50, MAP_SIZE-50), random.uniform(50, MAP_SIZE-50), val, col))

class RacingMode(GameMode):
    def initialize(self):
        self.name = "Racing"
        self.world.objects = []
        self.world.food = []
        for i in range(8):
            angle = math.radians(i * 45)
            x = MAP_SIZE // 2 + math.cos(angle) * 700
            y = MAP_SIZE // 2 + math.sin(angle) * 700
            self.world.objects.append(Checkpoint(i, x, y, i))
        for b in self.world.bots:
            b.status["next_checkpoint"] = 0

    def update(self):
        for b in self.world.bots:
            next_cp_idx = b.status.get("next_checkpoint", 0)
            cp = None
            for obj in self.world.objects:
                if isinstance(obj, Checkpoint) and obj.sequence_num == next_cp_idx:
                    cp = obj; break
            if cp:
                dist = math.sqrt((b.x - cp.x)**2 + (b.y - cp.y)**2)
                if dist < (b.radius + cp.radius):
                    b.receive_reward(200)
                    b.status["next_checkpoint"] = (next_cp_idx + 1) % 8

    def calculate_reward(self, bot):
        next_cp_idx = bot.status.get("next_checkpoint", 0)
        cp = None
        for obj in self.world.objects:
            if isinstance(obj, Checkpoint) and obj.sequence_num == next_cp_idx:
                cp = obj; break
        if cp:
            dist = math.sqrt((bot.x - cp.x)**2 + (bot.y - cp.y)**2)
            return (1000 - dist) / 10000.0
        return 0

class PuzzleCooperationMode(GameMode):
    def initialize(self):
        self.name = "Puzzle Cooperation"
        self.world.objects = [
            PuzzleElement(1, MAP_SIZE//4, MAP_SIZE//2, "plate"),
            PuzzleElement(2, MAP_SIZE*3//4, MAP_SIZE//2, "plate")
        ]
        self.world.food = []

    def update(self):
        all_activated = True
        for obj in self.world.objects:
            if isinstance(obj, PuzzleElement):
                obj.activated = False
                for b in self.world.bots:
                    dist = math.sqrt((b.x - obj.x)**2 + (b.y - obj.y)**2)
                    if dist < (b.radius + obj.radius):
                        obj.activated = True; break
                if not obj.activated: all_activated = False

        if all_activated:
            for b in self.world.bots:
                b.receive_reward(10)

if __name__ == '__main__':
    # Global Models for Meta-Learner
    device = torch.device("cpu")
    encoder = SharedEncoder().to(device)
    crn = CRN().to(device)
    mspns = torch.nn.ModuleList([MSPN().to(device) for _ in range(10)])
    mcn = MCN().to(device)

    # Load weights if available
    if os.path.exists("saves/shared_encoder_final.pth"):
        encoder.load_state_dict(torch.load("saves/shared_encoder_final.pth", map_location=device))
    if os.path.exists("saves/crn_model.pth"):
        crn.load_state_dict(torch.load("saves/crn_model.pth", map_location=device))
    if os.path.exists("saves/mspns_model.pth"):
        mspns.load_state_dict(torch.load("saves/mspns_model.pth", map_location=device))
    if os.path.exists("saves/mcn_model.pth"):
        mcn.load_state_dict(torch.load("saves/mcn_model.pth", map_location=device))

    bt = [RandomBot, RuleBasedBot, PotentialFieldBot, PIDControllerBot, GeneticBot, TabularQLearningBot, DeepQBot, ActorCriticBot, HeuristicSearchBot, EnsembleBot, NovelBot]
    for i, c in enumerate(bt):
        for j in range(2): world.bots.append(c(len(world.bots)))

    # Add Meta-Learners
    for i in range(3):
        world.bots.append(MetaBot(len(world.bots), encoder, crn, mspns, mcn))

    world.set_mode(ClassicArena)
    socketio.start_background_task(game_loop)
    socketio.run(app, host='0.0.0.0', port=5000)
