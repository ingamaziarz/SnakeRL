import random
import numpy as np

SIZE = 20
PIXEL = 20
WINDOW_SIZE = SIZE * PIXEL
FPS = 5
UP = (0, 1)
RIGHT = (1, 0)
DOWN = (0, -1)
LEFT = (-1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
ACTIONS = [-1, 0, 1] #-1: 90 degrees anticlockwise, 0: no change, 1: 90 degrees clockwise
#colors:
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Snake:
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.length = 3
        self.location = [(SIZE//2, SIZE//2), (SIZE//2, SIZE//2 + 1), (SIZE//2, SIZE//2 + 2)]
        self.direction = random.choice(DIRECTIONS)
        self.place_food()

    def head(self):
        return self.location[0]

    def step(self, action):
        reward = 0
        info = {}
        self.done = False
        self.direction = DIRECTIONS[(DIRECTIONS.index(self.direction) + action) % 4]
        self.done = self.move()
        if not self.done:
            if self.check_eaten():
                reward = 1
                info['food'] = True
        state = self.get_observation()
        return state, reward, self.done, False, info


    def move(self):
        new_head = tuple(map(lambda x, y: x + y, self.head(), self.direction))
        if new_head in self.location or any(x < 0 or x >= SIZE for x in new_head):
            self.initialize()
            return True
        self.location.insert(0, new_head)
        if self.head() != self.food:
            self.location.pop()
        return False

    def place_food(self):
        self.food = (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))
        if self.food in self.location:
            self.place_food()

    def check_eaten(self):
        if self.head() == self.food:
            self.length += 1
            self.place_food()
            return True
        return False

    def get_observation(self):
        screen = np.zeros((SIZE, SIZE))
        for s in self.location: screen[s] = 1
        screen[self.food] = 2
        return screen



