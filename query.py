import sys 
import pickle
import numpy as np
import dataformat
file_path = sys.argv[1]
file_des=open(file_path,'r')
data_list=file_des.readlines() #list containing data points
data = ' '.join(data_list)
#reading global feature vector 
gfv_desc=open('global_feature_vector','rb')
gfv = pickle.load(gfv_desc)
gfv_desc.close()
#loading U, S and V
try:
 U_desc=open('U_lowrank.pkl','rb')
except IOError as e:
 print str(e)
U = pickle.load(U_desc)
#U_desc.close()
S_desc=open('S_lowrank.pkl','rb')
S = pickle.load(S_desc)
V_desc=open('V_lowrank.pkl','rb')
V = pickle.load(V_desc)

def transformToGFV(reviewFeatureVector,gfeaturelist):
 unique_words=set(reviewFeatureVector)
 g_reviewfeaturedic=[]
 val=0
 for word in gfeaturelist:
  if word in unique_words: val=1
  else: val=0
  g_reviewfeaturedic.append(val)
 return g_reviewfeaturedic

def cosine_distance(u, v):
    """
    Returns the cosine of the angle between vectors v and u. This is equal to
    u.v / |u||v|.
    """
    return numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v)))

processed_data=dataformat.process_review(data)
fvector=dataformat.getfeaturevector(processed_data)
query_vector= np.array(transformToGFV(fvector,gfv))
#transforming query_vector to low rank space  1* k
S_inverse = np.linalg.inv(np.matrix(S))
query = np.matrix(np.dot(query_vector,V.T)) * S_inverse
max_sim=0
max_index=0
query=np.array(query)
for i in range(U.shape[0]):
 cos_sim = np.dot(query,U[i,])
 if cos_sim > max_sim:
  max_sim = cos_sim
  max_index=i

print max_sim 
print max_index

