from __future__ import absolute_import, division, print_function
import math
from inspect import isclass
from functools import wraps

__all__ = ['Box']

class hybridmethod(object):
    # http://stackoverflow.com/a/18078819/194586
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        context = obj if obj is not None else cls

        @wraps(self.func)
        def hybrid(*args, **kw):
            return self.func(context, *args, **kw)

        # optional, mimic methods some more
        hybrid.__func__ = hybrid.im_func = self.func
        hybrid.__self__ = hybrid.im_self = context

        return hybrid

class Box(object):
    def __init__(self, *args, **kwargs):
        """
        It's a box (rectangle, whatever).

        Box(left, right, bottom, top)
        Box([left, right, bottom, top])
        Box([left, right], [bottom, top])
        Box([[left, right], [bottom, top]])
        Box(left=left, right=right, bottom=bottom, top=top)
        Box(x=[left, right], y=[bottom, top])
        Box(center=[x, y], size=[width, height])
        Box.fit(points)

        Provides the following attributes to read/write: x, y, height, width,
        center, size.
        """
        self.left, self.right, self.bottom, self.top = 0, 0, 0, 0
        if not (len(args) or len(kwargs)):
            pass
        elif len(args) >= 1:
            if len(kwargs):
                raise ValueError('provide only positional or keyword arguments')

            if len(args) == 1:
                args = args[0]

            if len(args) == 2:
                (self.left, self.right), (self.bottom, self.top) = args
            elif len(args) == 4:
                self.left, self.right, self.bottom, self.top = args
            else:
                raise ValueError('unexpected positional argument format')

        elif (set(kwargs.keys()) == set(['left', 'right', 'bottom', 'top']) or
              set(kwargs.keys()) == set(['x', 'y']) or
              set(kwargs.keys()) == set(['center', 'size'])):
            for attr in kwargs:
                setattr(self, attr, kwargs[attr])

        else:
            raise ValueError('unexpected group of keyword arguments')

    def __len__(self):
        return 4

    def __repr__(self):
        return 'Box(left={0}, right={1}, bottom={2}, top={3})'.format(*self)

    def __add__(self, other):
        return self.__class__(
                min(self.left, other.left),
                max(self.right, other.right),
                min(self.bottom, other.bottom),
                max(self.top, other.top),
            )

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __eq__(self, other):
        return all(a == b for a, b in zip(self, other))
    def __ne__(self, other):
        return not self == other
    def __le__(self, other):
        return self + other == other
    def __ge__(self, other):
        return self + other == self
    def __lt__(self, other):
        return self <= other and all(s != o for s, o in zip(self, other))
    def __gt__(self, other):
        return self >= other and all(s != o for s, o in zip(self, other))

    def __contains__(self, point):
        x, y = point
        return (self.left <= x <= self.right) and (self.bottom <= y <= self.top)

    def __iter__(self):
        yield self.left
        yield self.right
        yield self.bottom
        yield self.top

    @hybridmethod
    def fit(hyb, points):
        # delegate to class or instance method as appropriate
        return hyb._fit_cls(points) if isclass(hyb) else hyb._fit_inst(points)

    @classmethod
    def _fit_cls(cls, points):
        ipoints = iter(points)
        xp, yp = next(ipoints)
        box = cls(xp, xp, yp, yp)
        box.fit(ipoints)
        return box

    def _fit_inst(self, points):
        for xp, yp in points:
            self.left = min(self.left, xp)
            self.right = max(self.right, xp)
            self.bottom = min(self.bottom, yp)
            self.top = max(self.top, yp)
        return self

    @property
    def xy(self):
        return self.x, self.y

    @property
    def bounds(self):
        return list(self)

    @property
    def width(self):
        return self.right - self.left
    @width.setter
    def width(self, dim):
        self.size = dim, self.height

    @property
    def height(self):
        return self.top - self.bottom
    @height.setter
    def height(self, dim):
        self.size = self.width, dim

    @property
    def x(self):
        return self.left, self.right
    @x.setter
    def x(self, lims):
        self.left, self.right = lims

    @property
    def y(self):
        return self.bottom, self.top
    @y.setter
    def y(self, lims):
        self.bottom, self.top = lims

    @property
    def center(self):
        return (self.right + self.left) / 2, (self.top + self.bottom) / 2
    @center.setter
    def center(self, point):
        hshift, vshift = (new - old for (new, old) in zip(point, self.center))
        self.x = (xcomp + hshift for xcomp in self.x)
        self.y = (ycomp + vshift for ycomp in self.y)

    @property
    def size(self):
        return self.right - self.left, self.top - self.bottom
    @size.setter
    def size(self, dims):
        width, height = dims
        xcent, ycent = self.center

        self.left = xcent - width / 2
        self.right = self.left + width

        self.bottom = ycent - height / 2
        self.top = self.bottom + height

    @property
    def T(self):
        """transpose"""
        return type(self)(x=self.y, y=self.x)

    def grow(self, delta):
        self.left -= delta
        self.bottom -= delta
        self.right += delta
        self.top += delta

    @property
    def PIL(self):
        return (
            int(math.floor(self.left)),
            int(math.floor(self.bottom)),
            int(math.ceil(self.right)),
            int(math.ceil(self.top)),
        )

    @property
    def vflip(self):
        return (
            self.left,
            self.right,
            self.top,
            self.bottom,
        )

    def round(self):
        for attr in ('left', 'right', 'top', 'bottom'):
            setattr(self, attr, int(round(getattr(self, attr))))

    def square(self, method='outer'):
        self.adjust_aspect(1, method)

    @property
    def aspect(self):
        return self.width / self.height

    def adjust_aspect(self, ratio, method='outer'):
        """Ratio is width over height"""
        if method == 'outer':
            if self.aspect < ratio:
                # expand width
                self.width = self.height * ratio
            else:
                # expand height
                self.height = self.width / ratio
        elif method == 'inner':
            if self.aspect < ratio:
                # shrink height
                self.height = self.width / ratio
            else:
                # shrink width
                self.width = self.height * ratio
        else:
            raise ValueError('Unknown method specified, use "inner" or "outer"')

def _tests():
    print('Crude tests...')

    def cmpall(one, two):
        return all(a == b for a, b in zip(one, two))

    # instancing
    Box() # should work
    b = Box(1, 2, 3, 4) # basic creation from 4
    assert b == Box([1, 2, 3, 4]), '1x4 creation broken'
    assert b == Box([1, 2], [3, 4]) , '2x2 creation broken'
    assert b == Box([[1, 2], [3, 4]]), '1x2x2 creation broken'
    assert b == Box(x=(1, 2), y=(3, 4)), 'x/y creation broken'
    assert b == Box(center=(1.5, 3.5), size=(1, 1)), 'center/size creation broken'
    assert b == Box(left=1, right=2, bottom=3, top=4), 'named bounds creation broken'

    # comparisons
    dims = 1, 2, 3, 4
    assert Box(dims) == Box(dims), '__eq__ broken'
    assert Box(dims) != Box(), '__ne__ broken'
    assert not (Box(dims) != Box(dims)), '__ne__ broken'
    assert Box(dims) < Box(0.9, 2.2, 2.9, 4.1), '__lt__ broken'
    assert not Box(dims) < Box(dims), '__lt__ broken'
    assert Box(dims) > Box(1.1, 1.9, 3.1, 3.9), '__gt__ broken'
    assert not Box(dims) > Box(dims), '__gt__ broken'
    assert Box(dims) <= Box(dims), '__le__ broken'
    assert not Box(dims) <= Box(1, 2, 3, 3.9), '__le__ broken'
    assert Box(dims) >= Box(dims), '__ge__ broken'
    assert not Box(dims) >= Box(1, 2, 3, 4.1), '__ge__ broken'

    # eval
    assert b == eval(repr(b)), '__eval__ broken'

    # iter
    assert cmpall(b, (1, 2, 3, 4)), '__iter__ broken'

    # getters
    assert cmpall(b.x, (1, 2)), 'x getter broken'
    assert cmpall(b.y, (3, 4)), 'y getter broken'
    assert cmpall(b.size, (1, 1)), 'size getter broken'
    assert cmpall(b.center, (1.5, 3.5)), 'center getter broken'

    # fit class/instance method
    assert b == Box.fit([(1, 3), (2, 4)]), 'creation via class method broken'
    b2 = Box.fit([(1, 3)])
    b2.fit([(2, 4)])
    assert b == b2, 'fitting via instance method broken'

    # setters
    b = Box()
    b.x = 5, 15
    assert cmpall(b, (5, 15, 0, 0)), 'x setter broken'
    b.size = 4, 4
    assert cmpall(b, (8, 12, -2, 2)), 'size setter broken'
    b.y = 20, 30
    assert cmpall(b, (8, 12, 20, 30)), 'y setter broken'
    b.center = -10, -20
    assert cmpall(b, (-12, -8, -25, -15)), 'center setter broken'

    # adding
    b1 = Box(x=(0, 4), y=(0, 4))
    b2 = Box(x=(2, 10), y=(2, 10))
    be = Box(0, 10, 0, 10)
    assert b1 + b2 == be, 'basic adding broken'
    assert sum([b1, b2]) == be, 'sum builtin broken'

    # transposition
    b = Box(x=(0, 10), y=(0, 5))
    assert b.T == Box(x=(0, 5), y=(0, 10)), 'transposition broken'

    # width/height
    b = Box(x=(0, 10), y=(0, 5))
    assert b.width == 10, 'width getter broken'
    assert b.height == 5, 'height getter broken'
    b.width = 20
    assert cmpall(b, (-5, 15, 0, 5)), 'width setter broken'
    b.height = 7
    assert cmpall(b, (-5, 15, -1, 6)), 'height setter broken'

    # growing
    b = Box(x=(0, 10), y=(0, 5))
    b.grow(10)
    assert cmpall(b, (-10, 20, -10, 15)), 'grow broken'

    # rounding
    b = Box(x=(0.1, 10.4), y=(0.6, 10.9))
    b.round()
    assert cmpall(b, (0, 10, 1, 11))

    # squaring/aspects
    b = Box(x=(0, 20), y=(0, 10))
    b.square()
    assert cmpall(b, (0, 20, -5, 15))

    b = Box(x=(0, 10), y=(0, 20))
    b.square('inner')
    assert cmpall(b, (0, 10, 5, 15))

    b = Box(x=(0, 10), y=(0, 20))
    b.adjust_aspect(3/2)
    assert cmpall(b, (-10, 20, 0, 20))

    b = Box(x=(0, 10), y=(0, 20))
    b.adjust_aspect(6/20, 'inner')
    assert cmpall(b, (2, 8, 0, 20))

    print('OK!')

if __name__ == '__main__':
    _tests()
