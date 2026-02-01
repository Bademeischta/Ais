# bots.py
import random
import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from entities import BaseBot, MAP_SIZE
from networks import DQNet, ACNet, SynergyNet, DuelingDQNet

class NovelBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id); self.algo_name = "Novel"; self.color = "#FFD700"
        self.device = device
        self.syn = SynergyNet().to(device); self.opt = torch.optim.Adam(self.syn.parameters(), lr=0.001)
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
        self.w1 = np.random.randn(228, 64) * 0.1; self.w2 = np.random.randn(64, 12) * 0.1; self.gen = 0
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

class DeepQBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id); self.algo_name = "DQN"; self.color = "#FF0000"
        self.device = device
        self.model = DQNet().to(device); self.target = DQNet().to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.0005); self.mem = deque(maxlen=50000)
        self.eps, self.steps, self.ls, self.la = 1.0, 0, None, None
    def decide(self, i):
        self.steps += 1; self.ls = i
        if random.random() < self.eps: self.la = random.randint(0, 11)
        else:
            with torch.no_grad(): self.la = torch.argmax(self.model(torch.FloatTensor(i).to(self.device))).item()
        return self.la
    def learn(self, r, ni, d):
        if self.ls is None: return
        self.mem.append((self.ls, self.la, r, ni, d))
        if len(self.mem) > 128 and self.steps % 4 == 0:
            import random
            b = random.sample(self.mem, 128); s, a, r, ns, d = zip(*b)
            s = torch.FloatTensor(np.array(s)).to(self.device)
            a = torch.LongTensor(a).to(self.device)
            r = torch.FloatTensor(r).to(self.device)
            ns = torch.FloatTensor(np.array(ns)).to(self.device)
            d = torch.BoolTensor(d).to(self.device)
            cq = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
            nq = self.target(ns).max(1)[0]; nq[d] = 0; tq = r + 0.99 * nq
            loss = torch.nn.MSELoss()(cq, tq.detach()); self.opt.zero_grad(); loss.backward(); self.opt.step()
            self.eps = max(0.05, self.eps * 0.9999)
            if self.steps % 500 == 0: self.target.load_state_dict(self.model.state_dict())
    def get_internal_metric(self): return f"E: {self.eps:.2f}"
    def save_state(self): return {'model': self.model.state_dict(), 'eps': self.eps}
    def load_state(self, s): self.model.load_state_dict(s['model']); self.eps = s['eps']

class ActorCriticBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id); self.algo_name = "A2C"; self.color = "#8B00FF"
        self.device = device
        self.model = ACNet().to(device); self.opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.llp, self.lv = None, None
    def decide(self, i):
        p, v = self.model(torch.FloatTensor(i).to(self.device)); d = torch.distributions.Categorical(p); a = d.sample()
        self.llp, self.lv = d.log_prob(a), v; return a.item()
    def learn(self, r, ni, d):
        if self.llp is None: return
        _, nv = self.model(torch.FloatTensor(ni).to(self.device)); t = r + (0 if d else 0.99 * nv.item())
        adv = t - self.lv.item()
        loss = -self.llp * adv + 0.5 * torch.nn.MSELoss()(self.lv.squeeze(), torch.FloatTensor([t]).to(self.device))
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


class CuriosityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(228 + 12, 128), nn.ReLU(),
            nn.Linear(128, 228)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(228 + 228, 128), nn.ReLU(),
            nn.Linear(128, 12)
        )

    def forward(self, state, action_oh, next_state):
        pred_next = self.forward_model(torch.cat([state, action_oh], dim=-1))
        pred_action = self.inverse_model(torch.cat([state, next_state], dim=-1))
        return pred_next, pred_action


class LSTMPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(228, 128, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 12), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, hidden=None):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        return self.actor(last_out), self.critic(last_out), hidden


class PPOBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id)
        self.algo_name = "PPO"
        self.color = "#FF1493"
        self.device = device
        self.policy = ACNet().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.buffer = []
        self.batch_size = 64
        self.ppo_epochs = 4
        self.clip_epsilon = 0.2

    def decide(self, inputs):
        with torch.no_grad():
            probs, value = self.policy(torch.FloatTensor(inputs).to(self.device))
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            self.current_state = inputs
            self.current_action = action.item()
            self.current_log_prob = dist.log_prob(action).item()
            self.current_value = value.item()
            return self.current_action

    def learn(self, reward, next_inputs, done):
        if not hasattr(self, "current_state"):
            return
        self.buffer.append((
            self.current_state,
            self.current_action,
            reward,
            self.current_log_prob,
            self.current_value,
            done,
        ))
        if len(self.buffer) >= self.batch_size:
            self._ppo_update()
            self.buffer = []

    def _ppo_update(self):
        states, actions, rewards, old_log_probs, old_values, dones = zip(*self.buffer)
        returns = []
        G = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                G = 0
            G = r + 0.99 * G
            returns.insert(0, G)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.ppo_epochs):
            probs, values = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

    def get_internal_metric(self):
        return f"Buf: {len(self.buffer)}"

    def save_state(self):
        return {"policy": self.policy.state_dict()}

    def load_state(self, state):
        self.policy.load_state_dict(state["policy"])


class DuelingDQNBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id)
        self.algo_name = "D-DQN"
        self.color = "#9400D3"
        self.device = device
        self.model = DuelingDQNet().to(device)
        self.target = DuelingDQNet().to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0003)
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.steps = 0
        self.ls, self.la = None, None

    def decide(self, inputs):
        self.steps += 1
        self.ls = inputs
        if random.random() < self.epsilon:
            self.la = random.randint(0, 11)
        else:
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(inputs).to(self.device))
                self.la = torch.argmax(q_values).item()
        return self.la

    def learn(self, reward, next_inputs, done):
        if self.ls is None:
            return
        self.memory.append((self.ls, self.la, reward, next_inputs, done))
        if len(self.memory) > 256 and self.steps % 4 == 0:
            batch = random.sample(self.memory, 256)
            s, a, r, ns, d = zip(*batch)
            s = torch.FloatTensor(np.array(s)).to(self.device)
            a = torch.LongTensor(a).to(self.device)
            r = torch.FloatTensor(r).to(self.device)
            ns = torch.FloatTensor(np.array(ns)).to(self.device)
            d = torch.BoolTensor(d).to(self.device)
            current_q = self.model(s).gather(1, a.unsqueeze(1)).squeeze()
            next_actions = self.model(ns).argmax(1)
            next_q = self.target(ns).gather(1, next_actions.unsqueeze(1)).squeeze()
            next_q[d] = 0
            target_q = r + 0.99 * next_q
            loss = nn.MSELoss()(current_q, target_q.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.epsilon = max(0.01, self.epsilon * 0.9995)
            if self.steps % 1000 == 0:
                self.target.load_state_dict(self.model.state_dict())

    def get_internal_metric(self):
        return f"ε: {self.epsilon:.2f}"

    def save_state(self):
        return {"model": self.model.state_dict(), "eps": self.epsilon}

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.epsilon = state["eps"]


class SACBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id)
        self.algo_name = "SAC"
        self.color = "#20B2AA"
        self.device = device
        self.actor = ACNet().to(device)
        self.critic1 = nn.Sequential(
            nn.Linear(228, 256), nn.ReLU(),
            nn.Linear(256, 12)
        ).to(device)
        self.critic2 = nn.Sequential(
            nn.Linear(228, 256), nn.ReLU(),
            nn.Linear(256, 12)
        ).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4
        )
        self.alpha = 0.2
        self.memory = deque(maxlen=50000)
        self.ls, self.la = None, None

    def decide(self, inputs):
        self.ls = inputs
        with torch.no_grad():
            probs, _ = self.actor(torch.FloatTensor(inputs).to(self.device))
            dist = torch.distributions.Categorical(probs)
            self.la = dist.sample().item()
        return self.la

    def learn(self, reward, next_inputs, done):
        if self.ls is None:
            return
        self.memory.append((self.ls, self.la, reward, next_inputs, done))
        if len(self.memory) > 128:
            batch = random.sample(self.memory, 128)
            s, a, r, ns, d = zip(*batch)
            s = torch.FloatTensor(np.array(s)).to(self.device)
            a = torch.LongTensor(a).to(self.device)
            r = torch.FloatTensor(r).to(self.device)
            ns = torch.FloatTensor(np.array(ns)).to(self.device)
            d = torch.BoolTensor(d).to(self.device)
            with torch.no_grad():
                next_probs, _ = self.actor(ns)
                next_q1 = self.critic1(ns)
                next_q2 = self.critic2(ns)
                next_q = torch.min(next_q1, next_q2)
                next_v = (next_probs * (next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=1)
                next_v[d] = 0
                target_q = r + 0.99 * next_v
            current_q1 = self.critic1(s).gather(1, a.unsqueeze(1)).squeeze()
            current_q2 = self.critic2(s).gather(1, a.unsqueeze(1)).squeeze()
            critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            probs, _ = self.actor(s)
            q1 = self.critic1(s)
            q2 = self.critic2(s)
            q = torch.min(q1, q2)
            actor_loss = (probs * (self.alpha * torch.log(probs + 1e-8) - q)).sum(dim=1).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

    def get_internal_metric(self):
        return f"α: {self.alpha:.2f}"

    def save_state(self):
        return {"actor": self.actor.state_dict(), "c1": self.critic1.state_dict(), "c2": self.critic2.state_dict()}

    def load_state(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["c1"])
        self.critic2.load_state_dict(state["c2"])


class CuriosityBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id)
        self.algo_name = "Curiosity"
        self.color = "#FFB6C1"
        self.device = device
        self.policy = ACNet().to(device)
        self.curiosity = CuriosityNet().to(device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.curiosity_opt = torch.optim.Adam(self.curiosity.parameters(), lr=3e-4)
        self.memory = deque(maxlen=10000)
        self.ls, self.la = None, None

    def decide(self, inputs):
        self.ls = inputs
        with torch.no_grad():
            probs, _ = self.policy(torch.FloatTensor(inputs).to(self.device))
            dist = torch.distributions.Categorical(probs)
            self.la = dist.sample().item()
        return self.la

    def learn(self, reward, next_inputs, done):
        if self.ls is None:
            return
        self.memory.append((self.ls, self.la, reward, next_inputs, done))
        if len(self.memory) > 64:
            batch = random.sample(self.memory, 64)
            s, a, r, ns, d = zip(*batch)
            s_t = torch.FloatTensor(np.array(s)).to(self.device)
            a_t = torch.LongTensor(a).to(self.device)
            r_t = torch.FloatTensor(r).to(self.device)
            ns_t = torch.FloatTensor(np.array(ns)).to(self.device)
            a_oh = torch.zeros(len(a), 12, device=self.device)
            a_oh.scatter_(1, a_t.unsqueeze(1), 1)
            pred_next, pred_action = self.curiosity(s_t, a_oh, ns_t)
            forward_loss = nn.MSELoss()(pred_next, ns_t)
            intrinsic_scalar = forward_loss.detach()
            inverse_loss = nn.CrossEntropyLoss()(pred_action, a_t)
            curiosity_loss = forward_loss + inverse_loss
            self.curiosity_opt.zero_grad()
            curiosity_loss.backward()
            self.curiosity_opt.step()
            total_reward = r_t + 0.5 * intrinsic_scalar
            probs, values = self.policy(s_t)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(a_t)
            policy_loss = -(log_probs * total_reward).mean()
            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

    def get_internal_metric(self):
        return f"Mem: {len(self.memory)}"

    def save_state(self):
        return {"policy": self.policy.state_dict(), "curiosity": self.curiosity.state_dict()}

    def load_state(self, state):
        self.policy.load_state_dict(state["policy"])
        self.curiosity.load_state_dict(state["curiosity"])


class LSTMBot(BaseBot):
    def __init__(self, bot_id, device=torch.device("cpu")):
        super().__init__(bot_id)
        self.algo_name = "LSTM"
        self.color = "#4169E1"
        self.device = device
        self.policy = LSTMPolicyNet().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.hidden = None
        self.memory = deque(maxlen=20000)
        self.sequence_buffer = deque(maxlen=20)
        self.ls, self.la = None, None

    def decide(self, inputs):
        self.ls = inputs
        self.sequence_buffer.append(inputs)
        with torch.no_grad():
            seq = torch.FloatTensor(np.array(list(self.sequence_buffer))).unsqueeze(0).to(self.device)
            probs, _, self.hidden = self.policy(seq, self.hidden)
            dist = torch.distributions.Categorical(probs)
            self.la = dist.sample().item()
        return self.la

    def learn(self, reward, next_inputs, done):
        if self.ls is None:
            return
        self.memory.append((list(self.sequence_buffer), self.la, reward, done))
        if done:
            self.hidden = None
        if len(self.memory) > 128:
            batch = random.sample(self.memory, 32)
            total_loss = 0
            losses = []
            for seq, a, r, d in batch:
                seq_t = torch.FloatTensor(np.array(seq)).unsqueeze(0).to(self.device)
                probs, value, _ = self.policy(seq_t)
                dist = torch.distributions.Categorical(probs)
                log_prob = dist.log_prob(torch.LongTensor([a]).to(self.device))
                advantage = r - value.item()
                actor_loss = -log_prob * advantage
                critic_loss = (advantage ** 2)
                loss = actor_loss + 0.5 * critic_loss
                losses.append(loss)
            if losses:
                avg_loss = torch.stack(losses).mean()
                self.optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

    def get_internal_metric(self):
        return f"Seq: {len(self.sequence_buffer)}"

    def save_state(self):
        return {"policy": self.policy.state_dict()}

    def load_state(self, state):
        self.policy.load_state_dict(state["policy"])


class MetaBot(BaseBot):
    """Meta-Learner: CRN erkennt Modus, MSPN liefert Policy; Confidence-Threshold für Exploration."""
    CONFIDENCE_THRESHOLD = 0.8  # Erst ab 80% Sicherheit MSPN nutzen, sonst Exploration
    CRN_UPDATE_INTERVAL = 20    # Alle 20 Frames CRN neu auswerten
    MIN_BUFFER_LEN = 50        # Mind. 50 Frames für CRN (Spec: 50–100)

    def __init__(self, bot_id, encoder, crn, mspns, mcn):
        super().__init__(bot_id)
        self.algo_name = "Meta-Learner"
        self.encoder = encoder
        self.crn = crn
        self.mspns = mspns  # ModuleList
        self.mcn = mcn
        self.obs_buffer = deque(maxlen=100)
        self.mode_probs = torch.zeros(10)
        self.active_mode_idx = 0
        self.confidence = 0.0
        self.uncertainty = 1.0

    def decide(self, inputs):
        self.obs_buffer.append(inputs)
        device = next(self.encoder.parameters()).device
        st = torch.FloatTensor(inputs).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = self.encoder(st)
            buf_len = len(self.obs_buffer)
            if buf_len >= self.MIN_BUFFER_LEN and buf_len % self.CRN_UPDATE_INTERVAL == 0:
                seq = torch.FloatTensor(np.array(list(self.obs_buffer))).unsqueeze(0).to(device)
                latent_seq = self.encoder(seq)
                crn_out = self.crn(latent_seq, return_uncertainty=True)
                if isinstance(crn_out, tuple):
                    self.mode_probs, unc = crn_out
                    self.mode_probs = self.mode_probs.squeeze(0)
                    self.uncertainty = unc.squeeze(0).item() if torch.is_tensor(unc) else unc
                else:
                    self.mode_probs = crn_out.squeeze(0)
                    self.uncertainty = 1.0 - self.mode_probs.max().item()
                self.active_mode_idx = torch.argmax(self.mode_probs).item()
                self.confidence = self.mode_probs[self.active_mode_idx].item()

            if self.confidence >= self.CONFIDENCE_THRESHOLD:
                probs, _ = self.mspns[self.active_mode_idx](latent)
                dist = torch.distributions.Categorical(probs)
                return dist.sample().item()
            # Exploration: gewichtete Mischung aus allen MSPN oder Random
            if random.random() < 0.3:
                return random.randint(0, 11)
            probs, _ = self.mspns[self.active_mode_idx](latent)
            dist = torch.distributions.Categorical(probs)
            return dist.sample().item()

    def get_internal_metric(self):
        modes = ["Classic", "Tag", "TDM", "CTF", "KotH", "BR", "Inf", "Res", "Race", "Puz"]
        if self.confidence < 0.3: return "Scanning..."
        return f"{modes[self.active_mode_idx]} ({self.confidence*100:.0f}%)"
