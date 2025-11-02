import jax.numpy as jnp
from jax import *
from utils import *



def points2Segments(points: jnp.ndarray)->jnp.ndarray:

    root = points[:3]
    atlas = points[3:6]
    pelvis = points[6:9]
    left_hip = points[9:12]
    right_hip = points[12:15]
    left_shoulder = points[15:18]
    right_shoulder = points[18:21]
    left_elbow = points[21:24]
    right_elbow = points[24:27]
    left_wrist = points[27:30]
    right_wrist = points[30:33]
    left_knee = points[33:36]
    right_knee = points[36:39]
    left_ankle = points[39:42]
    right_ankle = points[42:45]
    head = points[45:48]
    root2atlas = jnp.hstack([root,atlas])
    root2pelvis = jnp.hstack([root,pelvis])
    atlas2left_shoulder = jnp.hstack([atlas,left_shoulder])
    atlas2right_shoulder = jnp.hstack([atlas,right_shoulder])
    atlas2head = jnp.hstack([atlas,head])
    left_shoulder2left_elbow = jnp.hstack([left_shoulder,left_elbow])
    left_elbow2left_wrist =  jnp.hstack([left_elbow,left_wrist])
    right_shoulder2right_elbow = jnp.hstack([right_shoulder,right_elbow])
    right_elbow2right_wrist =  jnp.hstack([right_elbow,right_wrist])
    pelvis2left_hip = jnp.hstack([pelvis,left_hip])
    pelvis2right_hip = jnp.hstack([pelvis,right_hip])
    left_hip2left_knee = jnp.hstack([left_hip,left_knee])
    left_knee2left_ankle = jnp.hstack([left_knee,left_ankle])
    right_hip2right_knee = jnp.hstack([right_hip,right_knee])
    right_knee2right_ankle = jnp.hstack([right_knee,right_ankle])
    segments = jnp.vstack([ root2atlas,
                            root2pelvis,
                            atlas2left_shoulder,
                            atlas2right_shoulder,
                            atlas2head,
                            left_shoulder2left_elbow,
                            left_elbow2left_wrist,
                            right_shoulder2right_elbow,
                            right_elbow2right_wrist,
                            pelvis2left_hip,
                            pelvis2right_hip,
                            left_hip2left_knee,
                            left_knee2left_ankle,
                            right_hip2right_knee,
                            right_knee2right_ankle
                            ])
    
    return segments

def point2Segment_dist(point:jnp.ndarray, segment: jnp.ndarray)->float:
    diff = segment[3:] - segment[:3]
    alpha = jnp.clip(jnp.dot(point - segment[:3], diff)/(jnp.dot(diff,diff) + 1e-6),0,1)
    proj_point = segment[:3] + alpha * diff
    return jnp.linalg.norm(point - proj_point)

def point2Segments_dist(point:jnp.ndarray, segments: jnp.ndarray)->float:
    res = jnp.min(vmap(point2Segment_dist,(None,0))(point,segments))
    return res


def point2points_min_dist(point, points):
    dists = jnp.linalg.norm(points - point, axis=1)  # 计算point到points中每个点距离
    return jnp.min(dists)

# 估計點雲所有點到骨骼線段的距離的平方和
def points2Segments_dist_square_sum_loss(points:jnp.ndarray, segments: jnp.ndarray)->float:
    points2Segments_min_dist_vec = vmap(point2Segments_dist,(0,None))(points,segments)
    return  jnp.dot(points2Segments_min_dist_vec,points2Segments_min_dist_vec)

# 估計骨骼所有點到點雲距離的平方和
def points2points_dist_square_sum_loss(points1:jnp.ndarray, points2: jnp.ndarray)->float:
    points1 = jnp.reshape(points1,(-1,3))
    points2points_min_dist_vec = vmap(point2points_min_dist,(0,None))(points1,points2)
    return  jnp.dot(points2points_min_dist_vec,points2points_min_dist_vec)

def segments_equal_length_loss(segments: jnp.ndarray) ->float:
    diff = segments[:,3:] - segments[:,:3]
    diff_length = jnp.linalg.norm(diff,axis = 1)
    standard_diff_length = diff_length - jnp.mean(diff_length)
    loss = jnp.dot(standard_diff_length,standard_diff_length)
    return loss
def segments_symmetry_loss(segments: jnp.ndarray) -> float:
    diff = segments[:,3:] - segments[:,:3]
    diff_length = jnp.linalg.norm(diff,axis = 1)
    diff1 = diff_length[2] - diff_length[3]
    diff2 =  diff_length[5] - diff_length[7]
    diff3 =  diff_length[6] - diff_length[8]
    diff4 =  diff_length[9] - diff_length[10]
    diff5 =  diff_length[11] - diff_length[13]
    diff6 =  diff_length[12] - diff_length[14]
    return diff1**2. + diff2**2. + diff3**2. + diff4**2. + diff5**2. + diff6**2.

# def angle_loss(segments: jnp.ndarray) -> float:
#     diff = segments[:,3:] - segments[:,:3]
#     diff_length = jnp.linalg.norm(diff,axis = 1,keepdims=True)
#     normalize_segments = diff/(diff_length+1e-6)
#     loss = 0
#     # 把骨頭拉向T-Pose
#     # 左右鎖骨反向
#     atlas2left_atlas2right_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[3,:])
#     loss+= (atlas2left_atlas2right_angle_constraint - (-1.)) ** 2
    
#     # 左右盆骨反向
#     pelvis2left_pelvis2right_angle_constraint = jnp.dot(normalize_segments[9,:],normalize_segments[10,:])
#     loss+= (pelvis2left_pelvis2right_angle_constraint - (-1.)) ** 2
    
#     # 椎骨和鎖骨反向
#     root2atlas_root2pelvis2_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[1,:])
#     loss+= (root2atlas_root2pelvis2_angle_constraint - (-1.)) ** 2
    
#     # 鎖骨和盆骨方向平行
#     atlas2left_pelvis2left_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[9,:])
#     loss += (atlas2left_pelvis2left_angle_constraint - 1.) ** 2
    
#     # 椎骨和鎖骨垂直
#     root2atlas_atlas2left_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[2,:])
#     loss+= root2atlas_atlas2left_angle_constraint ** 2.
    
#     # 把頭部和椎骨對齊
#     root2atlas_atlas2head_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[4,:])
#     loss += (root2atlas_atlas2head_angle_constraint-1.)**2
    
#     # 把四根腿骨和椎骨反向對齊
#     root2atlas_left_hip2knee_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[11,:])
#     root2atlas_left_knee2ankle_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[12,:])
#     root2atlas_right_hip2knee_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[13,:])
#     root2atlas_right_knee2ankle_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[14,:])
#     loss += (root2atlas_left_hip2knee_angle_constraint - (-1)) ** 2 
#     loss += (root2atlas_left_knee2ankle_angle_constraint - (-1)) ** 2 
#     loss += (root2atlas_right_hip2knee_angle_constraint - (-1)) ** 2 
#     loss += (root2atlas_right_knee2ankle_angle_constraint - (-1)) ** 2 
    
    
#     # 讓四根手骨和鎖骨對齊
#     left_shoulder2elbow_atlas2left_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[5,:])
#     left_elbow2wrist_atlas2left_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[6,:])
#     right_shoulder2elbow_atlas2right_angle_constraint = jnp.dot(normalize_segments[3,:],normalize_segments[7,:])
#     right_elbow2wrist_atlas2right_angle_constraint = jnp.dot(normalize_segments[3,:],normalize_segments[8,:])
#     loss += (left_shoulder2elbow_atlas2left_angle_constraint - 1.) ** 2 
#     loss += (left_elbow2wrist_atlas2left_angle_constraint - 1.) ** 2 
#     loss += (right_shoulder2elbow_atlas2right_angle_constraint - 1.) ** 2 
#     loss += (right_elbow2wrist_atlas2right_angle_constraint - 1.) ** 2 
#     return loss


# # root           x,y,z  ,旋轉方向 
# # atlas          l1      旋轉方向: theta1,theta2,theta3
# # pelvis         l2      旋轉方向: 0, 0, theta4
# # left_shoulder  l3      旋轉方向: 0, 0, theta5
# # right_shoulder l3      旋轉方向: 0, 0, theta6
# # left_hip.      l4      旋轉方向: 0, 0, theta7
# # right_hip.     l4      旋轉方向: 0, 0, theta8
# # left_elbow.    l5      旋轉方向: theta9, theta10, theta11
# # right_elbow.   l5      旋轉方向: theta12, theta13, theta14
# # left_knee      l6      旋轉方向: theta15, theta16, theta17
# # right_knee     l6      旋轉方向: theta18, theta19, theta20
# # left_wrist.    l7      旋轉方向: 0, theta21, 0
# # right_wrist.   l7      旋轉方向: 0, theta22, 0
# # left_ankle.    l8      旋轉方向: 0, theta23, 0
# # right_ankle.   l8      旋轉方向: 0, theta24, 0
# # head.          l9      旋轉方向: theta25, theta26, theta27

# # x = params[0]
# # y = parma[1]
# # z = param[2]


def parms2Points(params:jnp.array)->jnp.array:
    l1,l2,l3,l4,l5,l6,l7,l8,l9 = params[3:12]
    theta1,theta2,theta3,theta4,theta5,theta6,theta7,theta8 = params[12:20]
    theta9,theta10,theta11,theta12,theta13,theta14,theta15,theta16 = params[20:28]
    theta17,theta18,theta19,theta20,theta21,theta22,theta23,theta24 = params[28:36]
    theta25,theta26,theta27 = params[36:39]   
   
    atlas_rotate_mat = create_rotate_matrix(jnp.array([theta1,theta2,theta3]))
    pelvis_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta4]))
    left_shoulder_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta5]))
    right_shoulder_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta6]))
    left_hip_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta7]))
    right_hip_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta8]))
    left_elbow_rotate_mat = create_rotate_matrix(jnp.array([theta9,theta10,theta11]))
    right_elbow_rotate_mat = create_rotate_matrix(jnp.array([theta12,theta13,theta14]))
    left_knee_rotate_mat = create_rotate_matrix(jnp.array([theta15,theta16,theta17]))
    right_knee_rotate_mat = create_rotate_matrix(jnp.array([theta18,theta19,theta20]))
    left_wrist_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta21]))
    right_wrist_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta22]))
    left_ankle_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta23]))
    right_ankle_rotate_mat = create_rotate_matrix(jnp.array([0.,0.,theta24]))
    head_rotate_mat = create_rotate_matrix(jnp.array([theta25,theta26,theta27]))
    end_dir = jnp.array([
       1.,0.,0.
    ]).reshape((-1,1))
    root = jnp.reshape(params[:3], (3, 1))
    atlas = root + l1 * atlas_rotate_mat @ end_dir
    pelvis = root + l2 * atlas_rotate_mat @ pelvis_rotate_mat @ end_dir

    left_hip = pelvis + l4 * atlas_rotate_mat @ pelvis_rotate_mat @ left_hip_rotate_mat @ end_dir
    right_hip = pelvis + l4 * atlas_rotate_mat @ pelvis_rotate_mat @ right_hip_rotate_mat @ end_dir

    left_shoulder = atlas + l3 * atlas_rotate_mat @ left_shoulder_rotate_mat @ end_dir
    right_shoulder = atlas + l3 * atlas_rotate_mat @ right_shoulder_rotate_mat @ end_dir

    left_elbow = left_shoulder + l5 * atlas_rotate_mat @ left_shoulder_rotate_mat @ left_elbow_rotate_mat @ end_dir
    right_elbow = right_shoulder + l5 * atlas_rotate_mat @ right_shoulder_rotate_mat @ right_elbow_rotate_mat @ end_dir

    left_wrist = left_elbow + l7 * atlas_rotate_mat @ left_shoulder_rotate_mat @ left_elbow_rotate_mat @ left_wrist_rotate_mat @ end_dir
    right_wrist = right_elbow + l7 * atlas_rotate_mat @ right_shoulder_rotate_mat @ right_elbow_rotate_mat @ right_wrist_rotate_mat @ end_dir

    left_knee = left_hip + l6 * atlas_rotate_mat @ pelvis_rotate_mat @ left_hip_rotate_mat @ left_knee_rotate_mat @ end_dir
    right_knee = right_hip + l6 * atlas_rotate_mat @ pelvis_rotate_mat @ right_hip_rotate_mat @ right_knee_rotate_mat @ end_dir

    left_ankle = left_knee + l8 * atlas_rotate_mat @ pelvis_rotate_mat @ left_hip_rotate_mat @ left_knee_rotate_mat @ left_ankle_rotate_mat @ end_dir
    right_ankle = right_knee + l8 * atlas_rotate_mat @ pelvis_rotate_mat @ right_hip_rotate_mat @ right_knee_rotate_mat @ right_ankle_rotate_mat @ end_dir

    head = atlas + l9 * atlas_rotate_mat @ head_rotate_mat @ end_dir
    
    res = jnp.hstack([
        jnp.reshape(root,(1,3)),
        jnp.reshape(atlas,(1,3)),
        jnp.reshape(pelvis,(1,3)),
        jnp.reshape(left_hip,(1,3)),
        jnp.reshape(right_hip,(1,3)),
        jnp.reshape(left_shoulder,(1,3)),
        jnp.reshape(right_shoulder,(1,3)),
        jnp.reshape(left_elbow,(1,3)),
        jnp.reshape(right_elbow,(1,3)),
        jnp.reshape(left_wrist,(1,3)),
        jnp.reshape(right_wrist,(1,3)),
        jnp.reshape(left_knee,(1,3)),
        jnp.reshape(right_knee,(1,3)),
        jnp.reshape(left_ankle,(1,3)),
        jnp.reshape(right_ankle,(1,3)),
        jnp.reshape(head,(1,3))
    ])
    return res

def positive_len_loss(params:jnp.ndarray)->float:
    lengths = params[3:12]
    lower_violation = jnp.maximum(0.5 - lengths, 0.0)
    upper_violation = jnp.maximum(lengths - 0.5, 0.0)
    return jnp.dot(lower_violation, lower_violation) + jnp.dot(upper_violation, upper_violation)
    


def angle_loss(params:jnp.ndarray) ->float:
    points = parms2Points(params)
    points = points[0]

    segments = points2Segments(points)

    diff = segments[:,3:] - segments[:,:3]
    diff_length = jnp.linalg.norm(diff,axis = 1,keepdims=True)
    normalize_segments = diff/(diff_length+1e-6)
    loss = 0

    # 左右鎖骨反向
    atlas2left_atlas2right_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[3,:])
    loss+= (atlas2left_atlas2right_angle_constraint - (-1.)) ** 2
    
    # 左右盆骨反向
    pelvis2left_pelvis2right_angle_constraint = jnp.dot(normalize_segments[9,:],normalize_segments[10,:])
    loss+=  (pelvis2left_pelvis2right_angle_constraint - (-1.)) ** 2
    
    # 椎骨和鎖骨反向
    root2atlas_root2pelvis2_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[1,:])
    loss+= 20 * (root2atlas_root2pelvis2_angle_constraint - (-1.)) ** 2
    
    # 鎖骨和盆骨方向平行
    atlas2left_pelvis2left_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[9,:])
    loss += (atlas2left_pelvis2left_angle_constraint - 1.) ** 2
    
    # 椎骨和鎖骨垂直
    root2atlas_atlas2left_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[2,:])
    loss+=  root2atlas_atlas2left_angle_constraint ** 2.
    
    # 把頭部和椎骨對齊
    root2atlas_atlas2head_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[4,:])
    loss += (root2atlas_atlas2head_angle_constraint-1.)**2
    
    # 把四根腿骨和椎骨反向對齊
    root2atlas_left_hip2knee_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[11,:])
    root2atlas_left_knee2ankle_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[12,:])
    root2atlas_right_hip2knee_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[13,:])
    root2atlas_right_knee2ankle_angle_constraint = jnp.dot(normalize_segments[0,:],normalize_segments[14,:])
    loss += jnp.minimum(root2atlas_left_hip2knee_angle_constraint,0.) ** 2 
    loss += jnp.minimum(root2atlas_left_knee2ankle_angle_constraint,0.) ** 2 
    loss += jnp.minimum(root2atlas_right_hip2knee_angle_constraint,0.) ** 2 
    loss += jnp.minimum(root2atlas_right_knee2ankle_angle_constraint,0.) ** 2 
    
    
    # 讓四根手骨和鎖骨對齊
    left_shoulder2elbow_atlas2left_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[5,:])
    left_elbow2wrist_atlas2left_angle_constraint = jnp.dot(normalize_segments[2,:],normalize_segments[6,:])
    right_shoulder2elbow_atlas2right_angle_constraint = jnp.dot(normalize_segments[3,:],normalize_segments[7,:])
    right_elbow2wrist_atlas2right_angle_constraint = jnp.dot(normalize_segments[3,:],normalize_segments[8,:])
    loss += jnp.minimum(left_shoulder2elbow_atlas2left_angle_constraint,0.) ** 2 
    loss += jnp.minimum(left_elbow2wrist_atlas2left_angle_constraint,0.) ** 2 
    loss += jnp.minimum(right_shoulder2elbow_atlas2right_angle_constraint,0.) ** 2 
    loss += jnp.minimum(right_elbow2wrist_atlas2right_angle_constraint,0.) ** 2 
    return loss

def theta_loss(params:jnp.ndarray)->float:
    thetas = params[12:39]
    
    # 限制角度在 [-pi, pi] 之間，超出懲罰
    upper_bound_violation = jnp.maximum(thetas - jnp.pi, 0)
    lower_bound_violation = jnp.maximum(-jnp.pi - thetas, 0)
    loss1 = jnp.sum(upper_bound_violation ** 2)
    loss2 = jnp.sum(lower_bound_violation ** 2)
    
    # wrist，ankle 關節限制在 [0, pi/2] 之間
    # theta0 = params[32:36] 對應 wrist, ankle 角度
    theta0 = params[32:36]
    wrist_ankle_upper_violation = jnp.maximum(theta0 - (jnp.pi / 2), 0)
    wrist_ankle_lower_violation = jnp.maximum(0 - theta0, 0)
    loss3 = jnp.sum(wrist_ankle_upper_violation ** 2)
    loss4 = jnp.sum(wrist_ankle_lower_violation ** 2)
    
    return loss1 + loss2 + loss3 + loss4
    

    