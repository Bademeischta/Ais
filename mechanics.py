# mechanics.py
import math
import numpy as np

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

class HeadlessArena:
    def __init__(self, mode_class, bots):
        from app import GameWorld
        self.world = GameWorld()
        self.world.bots = bots
        self.world.set_mode(mode_class)

    def step(self):
        from app import safe_decide
        experiences = []
        for b in self.world.bots:
            if b.died: continue
            old_state = b.get_input_vector(self.world)
            b.sense(self.world)
            action = safe_decide(b, old_state)
            b.apply_action(action)
            experiences.append((old_state, action, b))

        self.world.update()

        results = []
        for old_state, action, b in experiences:
            new_state = b.get_input_vector(self.world)
            reward = b.last_reward
            done = b.died
            results.append((old_state, action, reward, new_state, done))
            b.last_reward = 0

        return results, self.world.current_mode.check_victory()
