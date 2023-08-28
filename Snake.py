import random
import numpy as np


SIZE = 6
PIXEL = 20
FPS = 60
UP = (0, 1)
RIGHT = (1, 0)
DOWN = (0, -1)
LEFT = (-1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

class Snake:
    def __init__(self):
        self.initialize()
    def initialize(self):
        self.length = 3
        self.location = [(3, 3), (3, 4), (3, 5)] #TODO
        self.direction = random.choice(DIRECTIONS)
        self.place_food()
    def head(self):
        return self.location[0]
    def move(self):
        new = tuple(map(lambda x, y: x + y, self.head(), self.direction))
        if new in self.location or any(0 <= x < SIZE for x in new):
            self.initialize()
            raise SnakeException
        self.location.insert(0, new)
        self.location.pop()
    def place_food(self):
        self.food = (random.randint(0, SIZE - 1), random.randint(0, SIZE - 1))
        if self.food in self.location:
            self.place_food()

    def eaten(self):
        if self.head() == self.food:
            self.length += 1
            self.place_food()
class SnakeException(Exception):
    pass