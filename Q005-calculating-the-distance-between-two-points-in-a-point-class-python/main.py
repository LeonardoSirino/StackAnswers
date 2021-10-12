import math


class Point:
    """A 2D point in the cartesian plane"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Point({}, {})'.format(self._x, self._y)

    def dist_to_point(self, Point):
        dist = math.sqrt((self.x - Point.x)**2 + (self.y - Point.y)**2)
        return dist


p1 = Point(0, 0)
p2 = Point(1, 1)

distance = p1.dist_to_point(p2)

print(distance)
