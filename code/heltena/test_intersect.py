import euclid as eu

s1 = [1437.2242157960393, 1285.9535170711645]
s2 = [1437.974887431753, 1285.9535170711642]
p1 = [1416, 1291]
p2 = [1415, 1291]

s1 = eu.Point2(*s1)
s2 = eu.Point2(*s2)
ls = eu.Line2(s1, s2)

p1 = eu.Point2(*p1)
p2 = eu.Point2(*p2)
lp = eu.Line2(p1, p2)

print "S: ", s1, s2, ls
print "P: ", p1, p2, lp
print ls.intersect(lp)
