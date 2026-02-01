# app.py
# #region agent log
import sys, time, json as _dbg_json
try:
    _dbg_f = open(r'c:\Users\Administrator\Desktop\Ais\.cursor\debug.log', 'a'); _dbg_f.write(_dbg_json.dumps({"location":"app.py:entry","message":"app_started","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"H1","data":{}})+'\n'); _dbg_f.close()
except Exception: pass
# #endregion
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
from flask import Flask, render_template
from flask_socketio import SocketIO

from entities import MAP_SIZE, STARTING_MASS, FoodPellet, BaseBot
from mechanics import SpatialGrid
from networks import SharedEncoder, CRN, MSPN, MCN
from gamemodes import ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode, \
                      BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode
# Reihenfolge 0..9 für Frontend Modus-Auswahl und CRN
MODES = [ClassicArena, TagMode, TeamDeathmatchMode, CaptureTheFlagMode, KingOfTheHillMode,
         BattleRoyaleMode, InfectionMode, ResourceCollectorMode, RacingMode, PuzzleCooperationMode]
from bots import RandomBot, RuleBasedBot, PotentialFieldBot, PIDControllerBot, GeneticBot, \
                 TabularQLearningBot, DeepQBot, ActorCriticBot, HeuristicSearchBot, EnsembleBot, MetaBot, NovelBot
# #region agent log
try:
    _f = open(r'c:\Users\Administrator\Desktop\Ais\.cursor\debug.log', 'a'); _f.write(_dbg_json.dumps({"location":"app.py:imports","message":"imports_ok","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"H2","data":{}})+'\n'); _f.close()
except Exception: pass
# #endregion

# --- Configuration ---
FOOD_COUNT = 150
BOT_COUNT = 22
TICK_RATE = 30
MAX_BOT_MASS = 500
MIN_BOT_MASS = 15
SAVE_DIR = './saves/'

# Shared Lock
game_state_lock = threading.Lock()

def safe_decide(bot, inputs):
    try:
        inputs = np.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=0.0)
        action = bot.decide(inputs)
        if action is None or (isinstance(action, float) and np.isnan(action)):
            return 10
        return int(np.clip(action, 0, 11))
    except Exception as e:
        return 10

class GameWorld:
    def __init__(self):
        self.width, self.height = MAP_SIZE, MAP_SIZE
        self.bots = []
        self.objects = []
        self.food = []
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

        if self.current_mode:
            self.current_mode.update()

        for b in self.bots:
            if b.died: continue
            b.update_physics()

            m = b.radius
            if b.x < m: b.x, b.velocity[0] = m, 0; b.receive_reward(-3)
            elif b.x > self.width-m: b.x, b.velocity[0] = self.width-m, 0; b.receive_reward(-3)
            if b.y < m: b.y, b.velocity[1] = m, 0; b.receive_reward(-3)
            elif b.y > self.height-m: b.y, b.velocity[1] = self.height-m, 0; b.receive_reward(-3)

            if self.current_mode:
                b.receive_reward(self.current_mode.calculate_reward(b))

            for other in self.grid.get_nearby_entities(b.x, b.y):
                if b == other: continue
                dist = math.sqrt((b.x-other.x)**2 + (b.y-other.y)**2)
                if dist < b.radius:
                    if isinstance(other, BaseBot):
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

            b.mass = max(15, b.mass - b.mass * 0.0005)

    def respawn(self, b):
        b.receive_reward(-200); b.deaths += 1
        if self.current_mode and getattr(self.current_mode, "allow_respawn", True) == False:
            b.died = True; b.mass = 0; return
        b.x, b.y = random.uniform(100, self.width-100), random.uniform(100, self.height-100)
        b.mass, b.velocity = STARTING_MASS, [0.0, 0.0]

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
world = GameWorld()
# #region agent log
try:
    _f = open(r'c:\Users\Administrator\Desktop\Ais\.cursor\debug.log', 'a'); _f.write(_dbg_json.dumps({"location":"app.py:world","message":"world_created","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","data":{}})+'\n'); _f.close()
except Exception: pass
# #endregion

@app.route('/')
def index(): return render_template('index.html')


@socketio.on('set_mode')
def handle_set_mode(data):
    """Live-Modus-Wechsel: data = { mode_index: 0..9 }."""
    idx = data.get('mode_index', 0)
    if 0 <= idx < len(MODES):
        with game_state_lock:
            world.set_mode(MODES[idx])
        socketio.emit('mode_changed', {'mode': world.current_mode.name, 'mode_index': idx})


def game_loop():
    print("Loop started.")
    while True:
        st = time.time()
        with game_state_lock:
            for b in world.bots:
                if b.died: continue
                b.sense(world)
                a = safe_decide(b, b.get_input_vector(world))
                b.apply_action(a)
            world.update()
            for b in world.bots:
                if b.died: continue
                if hasattr(b, 'learn'):
                    iv = b.get_input_vector(world)
                    if isinstance(b, (TabularQLearningBot, GeneticBot)):
                        b.learn(b.last_reward, iv)
                    else:
                        b.learn(b.last_reward, iv, False)
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
                    if hasattr(b, 'ray_results'): d['rays'] = b.ray_results
                    if hasattr(b, 'confidence'): d['confidence'] = b.confidence
                    if hasattr(b, 'active_mode_idx'): d['active_mode_idx'] = b.active_mode_idx
                    bot_data.append(d)

                render_specs = world.current_mode.get_render_specs() if world.current_mode else {}
                socketio.emit('game_state', {
                    'frame': world.frame,
                    'mode': world.current_mode.name if world.current_mode else "None",
                    'mode_id': getattr(world.current_mode, 'mode_id', 0) if world.current_mode else 0,
                    'render_specs': render_specs,
                    'bots': bot_data,
                    'food': [{'x': f.x, 'y': f.y} for f in world.food],
                    'objects': [{'type': o.type, 'x': o.x, 'y': o.y, 'radius': o.radius, 'color': o.color, 'activated': getattr(o, 'activated', False)} for o in world.objects]
                })

        if world.frame % 1000 == 0:
            for b in world.bots:
                st_data = b.save_state()
                if st_data: torch.save(st_data, f"{SAVE_DIR}{b.name}.pth")

        time.sleep(max(0, 1/30 - (time.time() - st)))

if __name__ == '__main__':
    device = torch.device("cpu")
    encoder = SharedEncoder().to(device)
    crn = CRN().to(device)
    mspns = torch.nn.ModuleList([MSPN().to(device) for _ in range(10)])
    mcn = MCN().to(device)
    # #region agent log
    try:
        _f = open(r'c:\Users\Administrator\Desktop\Ais\.cursor\debug.log', 'a'); _f.write(_dbg_json.dumps({"location":"app.py:main","message":"networks_created","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"H3","data":{}})+'\n'); _f.close()
    except Exception: pass
    # #endregion

    # Load weights if available...

    bt = [RandomBot, RuleBasedBot, PotentialFieldBot, PIDControllerBot, GeneticBot, TabularQLearningBot, DeepQBot, ActorCriticBot, HeuristicSearchBot, EnsembleBot, NovelBot]

    for i, c in enumerate(bt):
        if c in [DeepQBot, ActorCriticBot, NovelBot]:
            world.bots.append(c(len(world.bots), device=device))
        else:
            world.bots.append(c(len(world.bots)))

    for i in range(3):
        world.bots.append(MetaBot(len(world.bots), encoder, crn, mspns, mcn))
    # #region agent log
    try:
        _f = open(r'c:\Users\Administrator\Desktop\Ais\.cursor\debug.log', 'a'); _f.write(_dbg_json.dumps({"location":"app.py:main","message":"bots_initialized","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"H4","data":{"n_bots":len(world.bots)}})+'\n'); _f.close()
    except Exception: pass
    # #endregion

    world.set_mode(ClassicArena)
    socketio.start_background_task(game_loop)
    # #region agent log
    try:
        _f = open(r'c:\Users\Administrator\Desktop\Ais\.cursor\debug.log', 'a'); _f.write(_dbg_json.dumps({"location":"app.py:main","message":"server_starting","timestamp":int(time.time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"H5","data":{"host":"127.0.0.1","port":5000}})+'\n'); _f.close()
    except Exception: pass
    # #endregion
    print("\n" + "=" * 50)
    print("  Ais Arena läuft lokal.")
    print("  Im Browser öffnen: http://localhost:5000")
    print("  Beenden: Strg+C")
    print("=" * 50 + "\n")
    socketio.run(app, host='127.0.0.1', port=5000)
