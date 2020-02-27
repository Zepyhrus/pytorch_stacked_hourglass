import numpy as np



def angle(v1, v2):
  """ Returns the angle in radians between vectors 'v1' and 'v2'::
    only works for 2d vector
  """
  def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    assert np.linalg.norm(vector) > 1e-6

    return vector / np.linalg.norm(vector)
  
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)

  return np.arctan2(v1[0]*v2[1]-v1[1]*v2[0], v1[0]*v2[0]+v1[1]*v2[1])

