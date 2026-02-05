
import random
import numpy as np

class DinoGame:
    def __init__(self):
        """
        State: [speed, obstacle_dist, obstacle_width, obstacle_height]
        """
        self.reset()
        
    def reset(self):
        self.speed = 10
        self.score = 0
        self.game_over = False
        self.time_alive = 0
        
        # Obstacle properties
        # For simplicity, one obstacle at a time
        self.obstacle_x = 1000 # Start far away
        self.obstacle_width = 50
        self.obstacle_height = 50
        self.obstacle_type = 0 # 0: none, 1: cactus
        
        # Player properties
        self.player_y = 0
        self.is_jumping = False
        self.jump_velocity = 0
        self.gravity = 2
        
        self._spawn_obstacle()
        return self.get_state()

    def _spawn_obstacle(self):
        self.obstacle_x = random.randint(800, 1200)
        self.obstacle_width = random.randint(20, 50)
        self.obstacle_height = random.randint(30, 70)
        self.obstacle_type = 1

    def step(self, action):
        """
        Action: 0 = Do Nothing, 1 = Jump
        """
        if self.game_over:
            return self.get_state(), 0, True

        reward = 1 # Survival reward
        
        # Apply action
        if action == 1 and not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = 25 # Initial jump strength

        # Physics Step
        if self.is_jumping:
            self.player_y += self.jump_velocity
            self.jump_velocity -= self.gravity
            
            if self.player_y <= 0:
                self.player_y = 0
                self.is_jumping = False
                self.jump_velocity = 0

        # Move obstacle
        self.obstacle_x -= self.speed
        
        # Spawn new obstacle
        if self.obstacle_x < -self.obstacle_width:
            self._spawn_obstacle()
            self.score += 10 # Bonus for clearing obstacle
            self.speed += 0.1 # Increase difficulty
            if self.speed > 25: self.speed = 25

        # Collision Detection
        # Simple box collision
        # Player box: x=0 to 40, y=player_y to player_y+40
        player_x = 0
        player_w = 40
        player_h = 40
        
        if (player_x < self.obstacle_x + self.obstacle_width and
            player_x + player_w > self.obstacle_x and
            self.player_y < self.obstacle_height and
            self.player_y + player_h > 0):
            
            self.game_over = True
            reward = -100
            
        self.time_alive += 1
        
        return self.get_state(), reward, self.game_over

    def get_state(self):
        # Normalized state validation for NN
        # [speed, obstacle_dist, obstacle_width, obstacle_height]
        # We normalize roughly to 0-1 range or at least consistent scale
        
        # dist can be up to 1200. /1200
        # width ~50. /100
        # height ~70. /100
        # speed ~20. / 30
        
        return np.array([
            self.speed, 
            self.obstacle_x,
            self.obstacle_width,
            self.obstacle_height
        ])
    
    def get_state_dict(self):
        """For visualization"""
        return {
            "speed": self.speed,
            "obstacle_x": self.obstacle_x,
            "obstacle_width": self.obstacle_width,
            "obstacle_height": self.obstacle_height,
            "player_y": self.player_y,
            "score": self.score
        }
