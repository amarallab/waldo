import scipy.misc
import scipy.ndimage.morphology as scimorph
import numpy as np
import math


class PlateDistance:
    def __init__(self, image):
        self.image = image
        self.lines = []      # list of four tuples: (m, b), where y = m * x + b defines a line
        self.tl = (0, 0)     # Top Left point
        self.tr = (0, 0)     # Top Right point
        self.bl = (0, 0)     # Bottom Left point
        self.br = (0, 0)     # Bottom Right point
        self.h1 = 0          # Horiz line size (top)
        self.h2 = 0          # Horiz line size (bottom)
        self.h = 0           # Horiz line size (average) 
        self.v1 = 0          # Vert line size (left)
        self.v2 = 0          # Vert line size (right)
        self.v = 0           # Vert lien size (average)
        self.aspect = 1      # Horiz / Vert aspect

    @staticmethod
    def __intersection_line_line(lines, a, b):
        m1, b1 = lines[a]
        m2, b2 = lines[b]
        if m1 == m2:
            return -1, -1
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y

    @staticmethod
    def __distance(a, b):
        x1, y1 = a
        x2, y2 = b
        x = x2 - x1
        y = y2 - y1
        return math.sqrt(x*x + y*y)

    def calculate(self):
        # Binarize
        n = self.image.copy()
        threshold = np.mean(n)
        n = n < threshold  # Working with booleans
        kernel = np.ones((30, 30))
        n = scimorph.binary_erosion(n, kernel)
        n = scimorph.binary_dilation(n, kernel)
    
        print("Shape: {} x {}".format(n.shape[0], n.shape[1]))
        center_x = n.shape[0] / 2
        center_y = n.shape[1] / 2

        arrows = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        lines = []
        for dx, dy in arrows:
            current_line = []
            for i in range(-300, 300, 2):
                if dx == 0:  # Looking top and bottom
                    x = center_x + i
                    y = center_y
                    n[x, 0] = True
                    n[x, n.shape[1] - 1] = True
                    while not n[x, y]:
                        y += dy
                else:
                    x = center_x
                    y = center_y + i
                    n[0, y] = True
                    n[n.shape[0] - 1, y] = True
                    while not n[x, y]:
                        x += dx
                current_line.append((x, y))
            xx, yy = zip(*current_line)
            m, b = np.polyfit(xx, yy, 1)
            lines.append((m, b))

        tl = self.__intersection_line_line(lines, 0, 2)
        tr = self.__intersection_line_line(lines, 0, 3)
        bl = self.__intersection_line_line(lines, 1, 2)
        br = self.__intersection_line_line(lines, 1, 3)

        h1 = self.__distance(tl, tr)
        h2 = self.__distance(bl, br)
        h = (h1 + h2) / 2

        v1 = self.__distance(tl, bl)
        v2 = self.__distance(tr, br)
        v = (v1 + v2) / 2

        aspect = h/v
        
        self.lines = lines
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br
        
        self.h1 = h1
        self.h2 = h2
        self.h = h

        self.v1 = v1
        self.v2 = v2
        self.v = v

        self.aspect = aspect
    
    def largest_distance(self):
        return self.h if self.h > self.v else self.v

    def largest_segment(self):
        print("TL {}, BL {}, TR {}, BR {}".format(self.tl, self.bl, self.tr, self.br))
        if self.h > self.v:
            p1 = [(self.tl[0] + self.bl[0]) / 2, (self.tl[1] + self.bl[1]) / 2]
            p2 = [(self.tr[0] + self.br[0]) / 2, (self.tr[1] + self.br[1]) / 2]
            return p1, p2
        else:
            p1 = [(self.tl[0] + self.tr[0]) / 2, (self.tl[1] + self.tr[1]) / 2]
            p2 = [(self.bl[0] + self.br[0]) / 2, (self.bl[1] + self.br[1]) / 2]
            return p1, p2

    def largest_segment_pretty(self, arrow_size=10):
        p1, p2 = self.largest_segment()
        v = (p2[0] - p1[0], p2[1] - p1[1])
        ang = math.atan2(v[1], v[0])
        
        r0 = p1[0] + arrow_size * math.cos(ang + math.pi/4), p1[1] + arrow_size * math.sin(ang + math.pi/4)
        r1 = p1[0] + arrow_size * math.cos(ang - math.pi/4), p1[1] + arrow_size * math.sin(ang - math.pi/4)

        r2 = p2[0] + arrow_size * math.cos(ang + math.pi*3.0/4.0), p2[1] + arrow_size * math.sin(ang + math.pi*3.0/4.0)
        r3 = p2[0] + arrow_size * math.cos(ang - math.pi*3.0/4.0), p2[1] + arrow_size * math.sin(ang - math.pi*3.0/4.0)
        return [[r0, p1, r1], [p1, p2], [r2, p2, r3]]

    def polygon(self, border, corner):
        if self.tl is None:
            return []

        # Left line
        n = self.tl[0] - self.bl[0], self.tl[1] - self.bl[1]
        d = math.hypot(n[0], n[1])
        if d == 0:
            return []
        n = n[0] / d, n[1] / d
        nt = n[1], -n[0]
        
        tlA = self.tl[0] + nt[0] * border - n[0] * corner, self.tl[1] + nt[1] * border - n[1] * corner
        blA = self.bl[0] + nt[0] * border + n[0] * corner, self.bl[1] + nt[1] * border + n[1] * corner

        # Right line
        n = self.tr[0] - self.br[0], self.tr[1] - self.br[1]
        d = math.hypot(n[0], n[1])
        if d == 0:
            return []
        n = n[0] / d, n[1] / d
        nt = -n[1], n[0]

        trA = self.tr[0] + nt[0] * border - n[0] * corner, self.tr[1] + nt[1] * border - n[1] * corner
        brA = self.br[0] + nt[0] * border + n[0] * corner, self.br[1] + nt[1] * border + n[1] * corner
    
        # Top line
        n = self.tl[0] - self.tr[0], self.tl[1] - self.tr[1]
        d = math.hypot(n[0], n[1])
        if d == 0:
            return []
        n = n[0] / d, n[1] / d
        nt = -n[1], n[0]

        tlB = self.tl[0] + nt[0] * border - n[0] * corner, self.tl[1] + nt[1] * border - n[1] * corner
        trB = self.tr[0] + nt[0] * border + n[0] * corner, self.tr[1] + nt[1] * border + n[1] * corner

        # Bottom line
        n = self.bl[0] - self.br[0], self.bl[1] - self.br[1]
        d = math.hypot(n[0], n[1])
        if d == 0:
            return []
        n = n[0] / d, n[1] / d
        nt = -n[1], n[0]

        blB = self.bl[0] - nt[0] * border - n[0] * corner, self.bl[1] - nt[1] * border - n[1] * corner
        brB = self.br[0] - nt[0] * border + n[0] * corner, self.br[1] - nt[1] * border + n[1] * corner
            
        return [tlA, tlB, trB, trA, brA, brB, blB, blA]        
        
    def __repr__(self):
        return "H side: {}, {}, mean: {}, V side: {}, {}, mean: {}, aspect: {}" \
            .format(self.h1, self.h2, self.h, self.v1, self.v2, self.v, self.aspect)