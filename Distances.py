def euclidean_distance(pt1, pt2):
    """ Find the euclidean distance between two points
    pt=[1,2,3,...,n] of the same dimention n"""
    distance = 0
    for i in range(len(pt1)):
        distance += (pt1[i] - pt2[i])**2
        return distance ** 0.5

def manhattan_distance(pt1, pt2):
    """ Find the manhattan distance between two points
    pt=[1,2,3,...,n] of the same dimention n"""
    distance = 0
    for i in range(len(pt1)):
      distance += abs(pt1[i] - pt2[i])
    return distance

def hamming_distance(pt1, pt2):
    """ Find the hamming distance between two points
    pt=[1,2,3,...,n] of the same dimention n"""
    distance = 0
    for i in range(len(pt1)):
      if pt1[i] == pt2[i]:
         distance += 0
      else:
        distance += 1
    return distance
