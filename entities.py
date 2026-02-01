# entities.py
import math
import random
import numpy as np

MAP_SIZE = 2000
STARTING_MASS = 25

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
        self.carrying_flag = False

    @property
    def radius(self):
        return math.sqrt(max(1, self.mass)) * 2.5

    def update_physics(self):
        self.time_alive_current += 1
        if self.mass > self.max_mass_achieved:
            self.max_mass_achieved = self.mass
        speed_mod = 5.0 / (1.0 + self.mass * 0.005)

        # Immediate mechanical effects for certain roles
        if self.team_id == 3: # Zombie or IT
            speed_mod *= 1.2

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

            # [dist, wall, food, friend_foe, is_bot, special, value, size, rel_mass]
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
                            res[4] = 1.0 # is_bot
                            if self.team_id != 0 and self.team_id == other.team_id:
                                res[3] = 1.0 # Friend
                            else:
                                res[3] = -1.0 # Foe
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
                            res[7] = getattr(other, 'mass', 10) / 500.0
            self.ray_results.append(res)

    def get_input_vector(self, world=None):
        inputs = []
        for r in self.ray_results:
            inputs.extend(r)

        # Target info for the mode
        target_dist, target_angle = 1.0, 0.0
        if world and world.current_mode:
            # Mode specific target identification
            target_pos = None
            if world.current_mode.name == "Capture the Flag":
                # Nearest enemy flag or own base
                if self.carrying_flag:
                    target_pos = (150, 150) if self.team_id == 1 else (MAP_SIZE-150, MAP_SIZE-150)
                else:
                    enemy_flag = next((obj for obj in world.objects if isinstance(obj, Flag) and obj.team_id != self.team_id), None)
                    if enemy_flag: target_pos = (enemy_flag.x, enemy_flag.y)
            elif world.current_mode.name == "Racing":
                next_cp_idx = self.status.get("next_checkpoint", 0)
                cp = next((obj for obj in world.objects if isinstance(obj, Checkpoint) and obj.sequence_num == next_cp_idx), None)
                if cp: target_pos = (cp.x, cp.y)
            elif world.current_mode.name == "King of the Hill":
                target_pos = (MAP_SIZE//2, MAP_SIZE//2)

            if target_pos:
                dx, dy = target_pos[0] - self.x, target_pos[1] - self.y
                target_dist = min(1.0, math.sqrt(dx**2 + dy**2) / 2000.0)
                target_angle = math.atan2(dy, dx) / math.pi

        inputs.extend([
            self.mass / 500.0, self.velocity[0], self.velocity[1],
            self.x / MAP_SIZE, self.y / MAP_SIZE, self.team_id / 5.0,
            1.0 if self.last_reward > 0 else (-1.0 if self.last_reward < 0 else 0.0),
            1.0 if self.died else 0.0,
            1.0 if self.status.get('is_it') or self.status.get('is_zombie') else 0.0,
            1.0 if self.carrying_flag else 0.0,
            target_dist, target_angle
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
