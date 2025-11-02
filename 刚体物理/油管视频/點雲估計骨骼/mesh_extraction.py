import numpy as np;
import scipy as sp;
import scipy.sparse as spsp;
from scipy.sparse.linalg import lsqr;
import trimesh

def getMeshVPos(mesh):
    vpos = []
    for v in mesh.vertices:
        vpos.append([v[0], v[1], v[2]])
    
 
    
    return np.array(vpos)

def meanCurvatureLaplaceWeights(mesh)->spsp.csr_matrix:
    rows = []
    cols = []
    data = []
    n = mesh.vertices.shape[0]
    for f in mesh.faces:
        v1 = mesh.vertices[f[0]]
        v2 = mesh.vertices[f[1]]
        v3 = mesh.vertices[f[2]]
        v1.index = f[0]
        v2.index = f[1]
        v3.index = f[2]
        v1v2 = v1 - v2
        v1v3 = v1 - v3
        
        v2v1 = v2 - v1
        v2v3 = v2 - v3
        
        v3v1 = v3 - v1
        v3v2 = v3 - v2            
        
        cot1 = np.dot(v2v1,v3v1) / max(np.linalg.norm(np.cross(v2v1, v3v1)),1e-6)
        cot2 = np.dot(v3v2,v1v2) / max(np.linalg.norm(np.cross(v3v2,v1v2)),1e-6)
        cot3 = np.dot(v1v3, v2v3) / max(np.linalg.norm(np.cross(v1v3,v2v3)),1e-6)
        
        rows.append(v2.index)
        cols.append(v3.index)
        data.append(cot1)
        
        rows.append(v3.index)
        cols.append(v2.index)
        data.append(cot1)
        
        rows.append(v3.index)
        cols.append(v1.index)
        data.append(cot2)
        
        rows.append(v1.index)
        cols.append(v3.index)
        data.append(cot2)
        
        rows.append(v1.index)
        cols.append(v2.index)
        data.append(cot3)
        
        rows.append(v2.index)
        cols.append(v1.index)
        data.append(cot3)           
    
    W = spsp.csr_matrix((data, (rows, cols)), shape = (n,n))
    sum_vector = W.sum(axis=0) 
    sum_vector_powered = np.power(sum_vector, -1.0)
    d = spsp.dia_matrix((sum_vector_powered, [0]), shape=(n,n))
    eye = spsp.identity(n)
    L = eye - d * W
    return L


def averageFaceArea(mesh:trimesh):

    area = []
    for f in mesh.faces:
        v1 = mesh.vertices[f[0]]
        v2 = mesh.vertices[f[1]]
        v3 = mesh.vertices[f[2]]
     
        area.append(0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1)))
 
    return 1.0 / (10.0 * np.sqrt(np.mean(area)))

def getOneRingAreas(mesh):
    oneringareas = []
    linked_face = {}
    for i,face in enumerate(mesh.faces):
        v1 = face[0]
        v2 = face[1]
        v3 = face[2]
        if(not v1 in linked_face): linked_face[v1] = set()
        if(not v2 in linked_face): linked_face[v2] = set()
        if(not v3 in linked_face): linked_face[v3] = set()
        linked_face[v1].add(i)
        linked_face[v2].add(i)
        linked_face[v3].add(i)

    
    for v in range(mesh.vertices.shape[0]):
        v_one_ring_area = []
        faces  = mesh.faces[list(linked_face[v])]
        for f in faces:
            v1 = mesh.vertices[f[0]]
            v2 = mesh.vertices[f[1]]
            v3 = mesh.vertices[f[2]]
            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
            v_one_ring_area.append(area)
        oneringareas.append(np.sqrt(np.sum(np.square(v_one_ring_area))))
    return np.array(oneringareas)


scene = trimesh.load('./scene.gltf')
mesh = trimesh.util.concatenate(scene)
cen = np.mean(mesh.vertices,axis =0)
gap = 0.6 * np.max(np.max(mesh.vertices,axis=0) - np.min(mesh.vertices,axis=0))




iterations = 7
n = mesh.vertices.shape[0]
SL = 10
initialFaceWeight = averageFaceArea(mesh)
originalOneRing = getOneRingAreas(mesh)
ovpos = getMeshVPos(mesh)
zeros = np.zeros((n,3))

np_WL0 = np.zeros((n,n))
np.fill_diagonal(np_WL0, initialFaceWeight)
WC = 10.0
WH0 = sp.sparse.dia_matrix(np.eye(n) * WC)
WL0 = sp.sparse.dia_matrix(np_WL0)


#dm.weighttype = 'Cotangents';
#bpy.ops.ashok.meshskeletonweightsoperator('EXEC_DEFAULT', currentobject=dm.name);
L = -meanCurvatureLaplaceWeights(mesh)
WL = sp.sparse.dia_matrix(WL0)
WH = sp.sparse.dia_matrix(WH0)




import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(iterations):
    print('ITERATION : %d'%i)
    vpos = mesh.vertices
    A = sp.sparse.vstack([L.dot(WL), WH])
    b = np.vstack((zeros, WH.dot(vpos)))
    cpts = np.zeros((n,3))
    for j in range(3):
        cpts[:, j] = lsqr(A, b[:, j])[0]
    mesh.vertices = cpts
    newringareas = getOneRingAreas(mesh) 
    changeinarea = np.power(newringareas, -0.5)
    WL = sp.sparse.dia_matrix(WL.multiply(SL))
    WH = sp.sparse.dia_matrix(WH0.multiply(changeinarea))
    L = -meanCurvatureLaplaceWeights(mesh)
    


ax.scatter(mesh.vertices[:,0],mesh.vertices[:,1],mesh.vertices[:,2], s= 1)
# 加上軸標籤（可選）
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([cen[0] -gap, cen[0] + gap])
ax.set_ylim([cen[1] -gap, cen[1] + gap])
ax.set_zlim([cen[2] -gap, cen[2] + gap])
ax.set_box_aspect([1, 1, 1])  # 顯示出來是等長
plt.show()
