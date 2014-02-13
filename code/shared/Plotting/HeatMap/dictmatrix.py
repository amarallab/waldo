#!/usr/bin/python
import sys, random

#
#     matrix represented by dictionaries of dictionaries.
#     This is appropriate for sparse matrices, and useful if you want to
#     label rows and columns by strings or other types instead of
#     integer indices.
#
#
#     author: Irmak Sirer
#


def mkmatrix(rows, cols, initialvalue):
    """ 
    Simple function to return a
    list of lists matrix filled
    with the initialvalue.
    """
    mx = [ None ] * rows
    for i in xrange(rows):
        mx[i] = [initialvalue] * cols
    return mx


class row(dict):
    def __init__(self, defaultvalue):
        self.defval = defaultvalue
        dict.__init__(self)
    def __getitem__(self,key):
        if key not in self:
            return self.defval
        else:
            return dict.__getitem__(self,key)    



class dictmatrix(dict):
    """
    matrix represented by dictionaries of dictionaries.
    This is appropriate for sparse matrices, and useful if you want to
    label rows and columns by strings or other types instead of
    integer indices.

    Very simple, does not have any matrix operations.
    """
    def __init__(self, defaultvalue):
        self.defval = defaultvalue
        dict.__init__(self)

    def __getitem__(self,key):
        if key not in self:
            return row(self.defval)
        else:
            return dict.__getitem__(self,key)
    def __repr__(self):
        if len(self.keys()) == 0:
            return "<empty_dictmatrix(defval=%s)>" % self.defval
        s = []
        for row in self.keys():
            for column in self[row]:
                s.append( "%6s\t" % self[row][column] )
            s.append("\n")
        return "".join(s)
            
    def put(self,i,j,value):
        if not self.has_key(i):
            self[i]=row(self.defval)
        self[i][j] = value

    def add(self,i,j,value):
        if not self.has_key(i):
            self[i]=row(self.defval)
        if not self[i].has_key(j):
            self[i][j] = value
        else:
            self[i][j] += value

    def remove(self,i,j):
        del self[i][j]
        if self[i]=={}:
            del self[i]

    def rows(self):
        return self.keys()

    def columns(self):
        cols = {}
        for row in self.keys():
            for col in self[row].keys():
                cols[col] = 1
        return cols.keys()

    def size(self):
        return len(set(self.rows()+self.columns()))

    def complete_rows_columns(self):
        rows, cols = self.rows(), self.columns()
        for row in rows:
            if row not in cols:
                self.put(rows[0],row,self.defval)
        for col in cols:
            if col not in rows:
                self.put(col,cols[0],self.defval)

    def exist(self,row,col):
        if row not in self: return False
        if col not in self[row]: return False
        return True

    def complete_square_matrix(self):
        self.complete_rows_columns()
        rows, cols = self.rows(), self.columns()
        for row in rows:
            for col in cols:
                if not self.exist(row,col):
                    self.put(row,col,self.defval)

    def allvalues(self):
        val = []
        for row in self.keys():
            for column in self[row]:
                val.append(self[row][column])
        return val

    def sum(self):
        return sum(self.allvalues())

    def __div__(self, number):
        """ only elemetwise dividing by a number """
        tmp = dictmatrix(self.defval)
        for row in self.keys():
            for column in self[row]:
                tmp.put(row,column, self[row][column]/number)
        return tmp
    
    def print_full_matrix(self, delimiter='\t',
                          complete_square=True, complete_to_square_size =None, 
                          shuffle_rows=False,
                          outputstream=sys.stdout):
        """square the matrix, print it """
        rows, cols = self.rows(), self.columns()
        # if row and col keys are supposed to be the same, do it
        if complete_square:
            if complete_to_square_size is not None:
                rows = range(complete_to_square_size)
                cols = rows[:]
            else:
                for col in cols:
                    if col not in rows:
                        rows.append(col)
                for row in rows:
                    if row not in cols:
                        cols.append(row)
        # dict has no order for keys,
        # regular sort is your best bet for ordering
        if shuffle_rows:
            args = range(len(rows))
            random.shuffle(args)
            transtable = dict(zip(range(len(rows)), args))
            newrows = []
            newcols = []
            for m in transtable:
                newrows.append(rows[m])
                newcols.append(cols[m])
            rows = newrows
            cols = newcols
        else:
            rows.sort(key=float)
            cols.sort(key=float)

        # template for row string
        Nc = len(cols)
        format = delimiter.join(['%g'] * Nc)

        # print every row
        for row in rows:
            vals = []
            for column in cols:
                vals.append(  self[row][column]  )
            
            print >> outputstream, format % tuple(vals)

    

if __name__ == '__main__':

    print "dictmatrix"
    mx = dictmatrix(0.)

    print "put something in mx[2][3]"
    mx.add(2,3,100.)

    print "read from mx"
    print mx[2][3]
    print mx[5][1]

    

