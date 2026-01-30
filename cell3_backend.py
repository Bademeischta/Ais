%%writefile app.py
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

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

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
        # print(f"⚠️ {bot.name} crashed during decision: {e}")
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

class FoodPellet:
    def __init__(self, fid, x, y):
        self.id = fid
        self.x = x
        self.y = y
        self.mass = 8.0

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
        self.ray_results = []
        self.device = device

    @property
    def radius(self):
        return math.sqrt(self.mass) * 2.5

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
        # Pre-filter nearby bots and food for efficiency
        nearby_entities = world.grid.get_nearby_entities(self.x, self.y)
        
        for i in range(self.vision_rays):
            angle = math.radians(i * (360 / self.vision_rays))
            dx, dy = math.cos(angle), math.sin(angle)
            res_dist, res_type, res_size = 1.0, 0.0, 0.0
            
            # Wall
            wall_dist = float('inf')
            if dx > 0: wall_dist = min(wall_dist, (world.width - self.x) / dx)
            elif dx < 0: wall_dist = min(wall_dist, -self.x / dx)
            if dy > 0: wall_dist = min(wall_dist, (world.height - self.y) / dy)
            elif dy < 0: wall_dist = min(wall_dist, -self.y / dy)
            if wall_dist < self.vision_range:
                res_dist, res_type = wall_dist / self.vision_range, 0.25
                
            # Food & Bots (using nearby entities)
            for other in nearby_entities:
                dist_sq = (other.x - self.x)**2 + (other.y - self.y)**2
                if dist_sq < self.vision_range**2:
                    dist = math.sqrt(dist_sq)
                    target_angle = math.atan2(other.y - self.y, other.x - self.x)
                    diff = abs((target_angle - angle + math.pi) % (2*math.pi) - math.pi)
                    
                    if diff < math.radians(7.5):
                        norm_dist = dist / self.vision_range
                        if norm_dist < res_dist:
                            res_dist = norm_dist
                            if isinstance(other, FoodPellet):
                                res_type = 0.50
                            else: # It's a bot
                                if other.mass > self.mass * 1.25: res_type = 1.0 # Threat
                                elif self.mass > other.mass * 1.25: res_type = 0.75 # Prey
                                else: res_type = 0.90 # Neutral
                                res_size = other.mass / 500
            self.ray_results.append((res_dist, res_type, res_size))

    def get_input_vector(self):
        inputs = []
        for r in self.ray_results: inputs.extend(r)
        # 8 additional inputs: mass, vel_x, vel_y, x, y, last_reward, bias1, bias2
        inputs.extend([self.mass/500, self.velocity[0], self.velocity[1], self.x/MAP_SIZE, self.y/MAP_SIZE, 
                       1.0 if self.last_reward > 0 else (-1.0 if self.last_reward < 0 else 0.0), 0.0, 0.0])
        return np.array(inputs, dtype=np.float32)

    def apply_action(self, action):
        if action < 8:
            a = math.radians(action * 45)
            self.velocity = [math.cos(a), math.sin(a)]
        elif action == 8: # BOOST
            self.velocity = [v * 1.8 for v in self.velocity]
            self.mass -= 0.05
        elif action == 9: # BRAKE
            self.velocity = [v * 0.5 for v in self.velocity]
        elif action == 10: # STAY
            self.velocity = [v * 0.9 for v in self.velocity]
        elif action == 11: # JUKE
            a = random.uniform(0, 2*math.pi)
            self.velocity = [math.cos(a), math.sin(a)]

    def decide(self, inputs): return 10
    def get_internal_metric(self): return ""
    def save_state(self): return {}
    def load_state(self, state): pass

# --- Existing Implementations (Adapted) ---

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
            d, t, s = inputs[i*3:i*3+3]
            if t == 1.0 and d < 0.2: self.st = "FLEE"; return (i * 15 + 180) // 45 % 8
            if t == 0.5 and d < 0.5: self.st = "FEED"; return (i * 15) // 45 % 8
        self.st = "EXP"; return 2

class PotentialFieldBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Field"; self.color = "#00FFFF"
    def decide(self, inputs):
        fx, fy = 0.0, 0.0
        for i in range(24):
            d, t, s = inputs[i*3:i*3+3]; a = math.radians(i*15); dx, dy = math.cos(a), math.sin(a)
            dr = d * 500 + 1
            if t == 0.5: fx += dx*50/(dr**1.5); fy += dy*50/(dr**1.5)
            elif t == 1.0: fx -= dx*500/(dr**2); fy -= dy*500/(dr**2)
            elif t == 0.25: fx -= dx*200/(dr**2); fy -= dy*200/(dr**2)
        if abs(fx) + abs(fy) < 0.01: return 10
        return int((math.degrees(math.atan2(fy, fx)) + 360) % 360 // 45) % 8

class PIDControllerBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "PID"; self.color = "#FFA500"
        self.integ, self.last_e = 0, 0
    def decide(self, inputs):
        ta = 0; min_d = 1.0
        for i in range(24):
            if inputs[i*3+1] == 0.5 and inputs[i*3] < min_d: min_d = inputs[i*3]; ta = i*15
        cur = math.degrees(math.atan2(self.velocity[1], self.velocity[0]))
        err = (ta - cur + 180) % 360 - 180
        self.integ += err; der = err - self.last_e; self.last_e = err
        out = 0.8 * err + 0.02 * self.integ + 0.3 * der
        return int((cur + out + 360) % 360 // 45) % 8

class GeneticBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Genetic"; self.color = "#00FF00"
        # Move to torch for GPU optimization
        self.w1 = torch.randn(80, 64, device=self.device) * 0.1
        self.w2 = torch.randn(64, 12, device=self.device) * 0.1
        self.gen = 0
    def decide(self, inputs):
        with torch.no_grad():
            x = torch.FloatTensor(inputs).to(self.device)
            h = torch.relu(x @ self.w1)
            l = h @ self.w2
            p = torch.softmax(l, dim=-1)
            return torch.multinomial(p, 1).item()
    def receive_reward(self, r):
        super().receive_reward(r)
        if r > 10 and random.random() < 0.2: self._mut(0.05)
        if r < -100: self._mut(0.2)
    def _mut(self, s): 
        self.w1 += torch.randn_like(self.w1) * s
        self.w2 += torch.randn_like(self.w2) * s
        self.gen += 1
    def get_internal_metric(self): return f"G: {self.gen}"
    def save_state(self): return {'w1': self.w1.cpu().numpy().tolist(), 'w2': self.w2.cpu().numpy().tolist(), 'gen': self.gen}
    def load_state(self, s): 
        self.w1 = torch.tensor(s['w1'], device=self.device)
        self.w2 = torch.tensor(s['w2'], device=self.device)
        self.gen = s['gen']

class TabularQLearningBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Q-Table"; self.color = "#FFFF00"
        self.q = {}; self.eps = 0.2; self.ls, self.la = None, None
    def _disc(self, i):
        mf, fa, mt, ta = 1.0, 0, 1.0, 0
        for r in range(24):
            if i[r*3+1] == 0.5 and i[r*3] < mf: mf, fa = i[r*3], r
            if i[r*3+1] == 1.0 and i[r*3] < mt: mt, ta = i[r*3], r
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

class DQNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(80, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 12))
    def forward(self, x): return self.net(x)

class DeepQBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "DeepQ (Legacy)"; self.color = "#FF4500"
        self.model = DQNet().to(self.device); self.target = DQNet().to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = optim.Adam(self.model.parameters(), lr=0.0005); self.mem = deque(maxlen=50000)
        self.eps, self.steps, self.ls, self.la = 1.0, 0, None, None
    def decide(self, i):
        self.steps += 1; self.ls = i
        if random.random() < self.eps: self.la = random.randint(0, 11)
        else: 
            with torch.no_grad(): 
                self.la = torch.argmax(self.model(torch.FloatTensor(i).to(self.device))).item()
        return self.la
    def learn(self, r, ni, d):
        if self.ls is None: return
        self.mem.append((self.ls, self.la, r, ni, d))
        if len(self.mem) > 128 and self.steps % 4 == 0:
            b = random.sample(self.mem, 128); s, a, r, ns, d = zip(*b)
            s = torch.FloatTensor(np.array(s)).to(self.device)
            a = torch.LongTensor(a).to(self.device)
            r = torch.FloatTensor(r).to(self.device)
            ns = torch.FloatTensor(np.array(ns)).to(self.device)
            d = torch.BoolTensor(d).to(self.device)
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
        super().__init__(); self.base = nn.Sequential(nn.Linear(80, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): b = self.base(x); return self.actor(b), self.critic(b)

class ActorCriticBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "A2C"; self.color = "#8B00FF"
        self.model = ACNet().to(self.device); self.opt = optim.Adam(self.model.parameters(), lr=0.001)
        self.llp, self.lv = None, None
    def decide(self, i):
        p, v = self.model(torch.FloatTensor(i).to(self.device)); d = torch.distributions.Categorical(p); a = d.sample()
        self.llp, self.lv = d.log_prob(a), v; return a.item()
    def learn(self, r, ni, d):
        if self.llp is None: return
        _, nv = self.model(torch.FloatTensor(ni).to(self.device)); t = r + (0 if d else 0.99 * nv.item())
        adv = t - self.lv.item()
        loss = -self.llp * adv + 0.5 * nn.MSELoss()(self.lv.squeeze(), torch.FloatTensor([t]).to(self.device))
        self.opt.zero_grad(); loss.backward(); self.opt.step()
    def save_state(self): return {'model': self.model.state_dict()}
    def load_state(self, s): self.model.load_state_dict(s['model'])

class HeuristicSearchBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Search"; self.color = "#0000FF"; self.sc = 0
    def decide(self, i):
        best, ms = 10, -999
        for a in [0, 2, 4, 6, 8, 10]:
            s = 0
            for r in range(24):
                d, t, sz = i[r*3:r*3+3]; ra = r*15
                if t == 0.5 and abs(ra - a*45) < 45: s += 10/(d+0.1)
                if t == 1.0 and abs(ra - a*45) < 45: s -= 50/(d+0.1)
            if s > ms: ms, best = s, a
        self.sc = ms; return best

class EnsembleBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Ensemble"; self.color = "#FF69B4"
    def decide(self, i):
        v1 = random.randint(0, 7); mf, fa = 1.0, 0
        for r in range(24):
            if i[r*3+1] == 0.5 and i[r*3] < mf: mf, fa = i[r*3], r
        v2 = fa//3; mt, ta = 1.0, 0
        for r in range(24):
            if i[r*3+1] == 1.0 and i[r*3] < mt: mt, ta = i[r*3], r
        v3 = (ta//3 + 4) % 8 if mt < 0.3 else v2
        c = {}; 
        for v in [v1, v2, v3]: c[v] = c.get(v, 0) + 1
        return max(c, key=c.get)

class SynergyNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Linear(80+12, 128), nn.ReLU(), nn.Linear(128, 12))
    def forward(self, s, a_oh): return self.net(torch.cat([s, a_oh], dim=-1))

class NovelBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Novel"; self.color = "#FFD700"
        self.syn = SynergyNet().to(self.device); self.opt = optim.Adam(self.syn.parameters(), lr=0.001)
        self.eps_mem = []; self.cur_traj = []; self.re_mode = False; self.re_idx = 0; self.active_ep = None
    def decide(self, i):
        st = torch.FloatTensor(i).to(self.device)
        if not self.re_mode and self.eps_mem:
            for ep in self.eps_mem:
                if np.linalg.norm(i - ep['s']) < 0.4: self.re_mode, self.active_ep, self.re_idx = True, ep, 0; break
        if self.re_mode:
            ba = self.active_ep['acts'][self.re_idx]; ba_oh = torch.zeros(12, device=self.device); ba_oh[ba] = 1.0
            with torch.no_grad(): off = self.syn(st, ba_oh)
            l = torch.zeros(12, device=self.device); l[ba] = 3.0; l += off
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
            if len(self.cur_traj) > 5:
                s, a, r_seq = zip(*self.cur_traj); s = torch.FloatTensor(np.array(s)).to(self.device)
                a_oh = torch.zeros(len(a), 12, device=self.device)
                for idx, act in enumerate(a): a_oh[idx, act] = 1.0
                out = self.syn(s, a_oh)
                loss = (out.pow(2)).mean() * 0.01
                if tr > 50: loss -= (out.sum()) * 0.001 
                self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.cur_traj = []
    def get_internal_metric(self): return f"EP: {len(self.eps_mem)}"
    def save_state(self): return {'syn': self.syn.state_dict(), 'eps': [{'s': e['s'].tolist(), 'acts': e['acts'], 'r': e['r']} for e in self.eps_mem]}
    def load_state(self, s): 
        self.syn.load_state_dict(s['syn'])
        self.eps_mem = [{'s': np.array(e['s']), 'acts': e['acts'], 'r': e['r']} for e in s['eps']]

# --- NEW LEARNERS ---

class DuelingDQNet(nn.Module):
    def __init__(self, input_dim=80, output_dim=12):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.advantage = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, output_dim))
        self.value = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

class DQNBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "DQN-Pro"; self.color = "#FF0000"
        self.model = DuelingDQNet().to(self.device); self.target = DuelingDQNet().to(self.device)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = optim.Adam(self.model.parameters(), lr=0.0003); self.mem = deque(maxlen=100000)
        self.eps, self.steps, self.ls, self.la = 1.0, 0, None, None
        self.batch_size = 128
        self.gamma = 0.99

    def decide(self, i):
        self.steps += 1; self.ls = i
        if random.random() < self.eps: self.la = random.randint(0, 11)
        else:
            with torch.no_grad():
                st = torch.FloatTensor(i).unsqueeze(0).to(self.device)
                self.la = torch.argmax(self.model(st)).item()
        return self.la

    def learn(self, r, ni, d):
        if self.ls is None: return
        self.mem.append((self.ls, self.la, r, ni, d))
        if len(self.mem) > self.batch_size:
            b = random.sample(self.mem, self.batch_size)
            s, a, r, ns, d = zip(*b)
            s = torch.FloatTensor(np.array(s)).to(self.device)
            a = torch.LongTensor(a).to(self.device)
            r = torch.FloatTensor(r).to(self.device)
            ns = torch.FloatTensor(np.array(ns)).to(self.device)
            d = torch.BoolTensor(d).to(self.device)

            q_values = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
            next_q_values = self.target(ns).max(1)[0]
            next_q_values[d] = 0.0
            expected_q_values = r + self.gamma * next_q_values

            loss = nn.MSELoss()(q_values, expected_q_values.detach())
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.eps = max(0.05, self.eps * 0.99995)
            if self.steps % 1000 == 0: self.target.load_state_dict(self.model.state_dict())

    def get_internal_metric(self): return f"E: {self.eps:.2f}"
    def save_state(self): return {'model': self.model.state_dict(), 'eps': self.eps}
    def load_state(self, s): self.model.load_state_dict(s['model']); self.eps = s['eps']

class PPOBot(BaseBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "PPO-Tactical"; self.color = "#00FF7F"
        self.actor = nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 12), nn.Softmax(dim=-1)).to(self.device)
        self.critic = nn.Sequential(nn.Linear(80, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)).to(self.device)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=0.001)
        self.mem = []; self.ls, self.la, self.lp = None, None, None

    def decide(self, i):
        self.ls = i
        with torch.no_grad():
            st = torch.FloatTensor(i).to(self.device)
            probs = self.actor(st)
            dist = torch.distributions.Categorical(probs)
            self.la = dist.sample()
            self.lp = dist.log_prob(self.la).item()
        return self.la.item()

    def learn(self, r, ni, d):
        if self.ls is None: return
        self.mem.append((self.ls, self.la, self.lp, r, ni, d))
        if len(self.mem) >= 64:
            s, a, lp, r, ns, d = zip(*self.mem); self.mem = []
            s = torch.FloatTensor(np.array(s)).to(self.device)
            a = torch.stack(a).to(self.device)
            lp = torch.FloatTensor(lp).to(self.device)
            r = torch.FloatTensor(r).to(self.device)
            ns = torch.FloatTensor(np.array(ns)).to(self.device)
            d = torch.BoolTensor(d).to(self.device)

            with torch.no_grad():
                target_v = r + 0.99 * self.critic(ns).squeeze() * (~d)
                adv = target_v - self.critic(s).squeeze()

            # PPO Update
            for _ in range(4):
                new_probs = self.actor(s)
                new_dist = torch.distributions.Categorical(new_probs)
                new_lp = new_dist.log_prob(a)
                ratio = torch.exp(new_lp - lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
                loss_a = -torch.min(surr1, surr2).mean()
                loss_c = nn.MSELoss()(self.critic(s).squeeze(), target_v)

                self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
                self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()

    def save_state(self): return {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}
    def load_state(self, s): self.actor.load_state_dict(s['actor']); self.critic.load_state_dict(s['critic'])

class SniperBot(PPOBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Sniper-PPO"; self.color = "#FFD700"
    
    def receive_reward(self, r):
        # Sniper reward shaping: Bonus for kills if they were "targeted"
        # We can use the vision info to see if we were looking at prey
        bonus = 0
        if r > 100: # Kill reward
            # Check if we were moving towards a target in previous frames (simplified)
            bonus = 50 
        super().receive_reward(r + bonus)

class SurvivalBot(PPOBot):
    def __init__(self, bot_id):
        super().__init__(bot_id); self.algo_name = "Survival-PPO"; self.color = "#00BFFF"
    
    def receive_reward(self, r):
        # Survival reward shaping:
        # 1. Constant survival reward
        # 2. Punishment for being near threats (from inputs)
        bonus = 0.5 
        if r < -10: # Death or wall hit
            bonus -= 5
        super().receive_reward(r + bonus)
    
    def decide(self, inputs):
        # Extra penalty for being near threats sensed in vision
        threat_penalty = 0
        for i in range(24):
            d, t, s = inputs[i*3:i*3+3]
            if t == 1.0 and d < 0.3: # Threat nearby
                threat_penalty -= (1.0 - d) * 2
        self.receive_reward(threat_penalty)
        return super().decide(inputs)

# --- HEADLESS SIMULATION ENGINE ---

class HeadlessArena:
    def __init__(self, bots, map_size=2000, food_count=100):
        self.width = map_size
        self.height = map_size
        self.bots = bots
        self.food = []
        self.grid = SpatialGrid(map_size)
        self.frame = 0
        self.max_frames = 1000 # Episode length
        for _ in range(food_count): self.spawn_food()

    def spawn_food(self):
        self.food.append(FoodPellet(random.randint(0, 10**9), random.uniform(20, self.width-20), random.uniform(20, self.height-20)))

    def step(self):
        self.frame += 1
        self.grid.clear()
        for b in self.bots: self.grid.add_entity(b)
        for f in self.food: self.grid.add_entity(f)
        
        for b in self.bots:
            b.update_physics()
            # Wall Collision
            m = b.radius
            if b.x < m: b.x, b.velocity[0] = m, 0; b.receive_reward(-2)
            elif b.x > self.width-m: b.x, b.velocity[0] = self.width-m, 0; b.receive_reward(-2)
            if b.y < m: b.y, b.velocity[1] = m, 0; b.receive_reward(-2)
            elif b.y > self.height-m: b.y, b.velocity[1] = self.height-m, 0; b.receive_reward(-2)
            
            # Food Eating
            for f in self.food[:]:
                dist_sq = (b.x-f.x)**2 + (b.y-f.y)**2
                if dist_sq < (b.radius + 5)**2:
                    b.mass = min(500, b.mass + 8); b.food_eaten += 1
                    b.receive_reward(10); self.food.remove(f); self.spawn_food()
            
            # Bot Combat
            nearby = self.grid.get_nearby_entities(b.x, b.y)
            for other in nearby:
                if b == other or not isinstance(other, BaseBot): continue
                dist_sq = (b.x-other.x)**2 + (b.y-other.y)**2
                if dist_sq < b.radius**2:
                    if b.mass > other.mass * 1.25:
                        b.mass = min(500, b.mass + other.mass * 0.8)
                        b.kills += 1; b.receive_reward(100)
                        self.respawn(other)
            
            # Passive Mass Decay
            b.mass = max(15, b.mass - b.mass * 0.0005)
        
        return self.frame >= self.max_frames

    def respawn(self, b):
        b.receive_reward(-50); b.deaths += 1
        b.x, b.y = random.uniform(100, self.width-100), random.uniform(100, self.height-100)
        b.mass, b.velocity = 25, [0.0, 0.0]

# --- World & Server (For Visualization) ---

class GameWorld:
    def __init__(self):
        self.width, self.height, self.bots, self.food, self.grid, self.frame = MAP_SIZE, MAP_SIZE, [], [], SpatialGrid(), 0
        for _ in range(FOOD_COUNT): self.spawn_food()
    def spawn_food(self): self.food.append(FoodPellet(random.randint(0,10**9), random.uniform(20, 1980), random.uniform(20, 1980)))
    def update(self):
        self.frame += 1; self.grid.clear()
        for b in self.bots: self.grid.add_entity(b)
        for f in self.food: self.grid.add_entity(f)
        for b in self.bots:
            b.update_physics()
            m = b.radius
            if b.x < m: b.x, b.velocity[0] = m, 0; b.receive_reward(-3)
            elif b.x > 2000-m: b.x, b.velocity[0] = 2000-m, 0; b.receive_reward(-3)
            if b.y < m: b.y, b.velocity[1] = m, 0; b.receive_reward(-3)
            elif b.y > 2000-m: b.y, b.velocity[1] = 2000-m, 0; b.receive_reward(-3)
            for f in self.food[:]:
                if math.sqrt((b.x-f.x)**2 + (b.y-f.y)**2) < b.radius + 5:
                    b.mass = min(500, b.mass+8); b.food_eaten += 1; b.receive_reward(15); self.food.remove(f); self.spawn_food()
            for other in self.grid.get_nearby_entities(b.x, b.y):
                if b == other: continue
                if math.sqrt((b.x-other.x)**2 + (other.y-b.y)**2) < b.radius:
                    if b.mass > other.mass*1.25: b.mass = min(500, b.mass+other.mass*0.8); b.kills+=1; b.receive_reward(150); self.respawn(other)
            b.mass = max(15, b.mass - b.mass*0.0005)
    def respawn(self, b):
        b.receive_reward(-200); b.deaths += 1; b.x, b.y, b.mass, b.velocity = random.uniform(100, 1900), random.uniform(100, 1900), 25, [0,0]

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
world = GameWorld()

@app.route('/')
def index(): return render_template('index.html')

def game_loop():
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
                    if isinstance(b, (TabularQLearningBot, GeneticBot)):
                        b.learn(b.last_reward, b.get_input_vector())
                    else:
                        b.learn(b.last_reward, b.get_input_vector(), False)
                b.last_reward = 0
            if world.frame % 2 == 0:
                socketio.emit('game_state', {'frame': world.frame, 'bots': [{'name': b.name, 'algo_name': b.algo_name, 'color': b.color, 'x': b.x, 'y': b.y, 'mass': b.mass, 'radius': b.radius, 'metric': b.get_internal_metric()} for b in world.bots], 'food': [{'x': f.x, 'y': f.y} for f in world.food]})
        if world.frame % 1000 == 0:
            for b in world.bots:
                st_data = b.save_state()
                if st_data: torch.save(st_data, f"{SAVE_DIR}{b.name}.pth")
        time.sleep(max(0, 1/30 - (time.time() - st)))

if __name__ == '__main__':
    bot_types = [RandomBot, RuleBasedBot, PotentialFieldBot, PIDControllerBot, GeneticBot, TabularQLearningBot, DeepQBot, ActorCriticBot, HeuristicSearchBot, EnsembleBot, NovelBot, DQNBot, PPOBot, SniperBot, SurvivalBot]
    for i, c in enumerate(bot_types):
        count = 2 if i < 11 else 1 # More of the older ones, 1 of each new one for the live view
        for j in range(count): world.bots.append(c(len(world.bots)))
    socketio.start_background_task(game_loop)
    socketio.run(app, host='0.0.0.0', port=5000)
