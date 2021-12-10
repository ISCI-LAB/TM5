#!/usr/bin/env python
# coding: utf-8

# In[3]:


""" GraspNet dataset processing.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
BASE_DIR = os.path.dirname('/home/po/TM5/graspnet-baseline/dataset/graspnet_dataset.py')
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join('/home/po/TM5/graspnet-baseline', 'utils'))
from data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,                            get_workspace_mask, remove_invisible_grasp_points
BASE_DIR


#graspnetAPI graspnet.py API interface Path
sys.path.append('/home/po/TM5/graspnetAPI/graspnetAPI')
sys.path.append('/home/po/TM5/graspnetAPI/graspnetAPI/utils')
from graspnetAPI.grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup, RECT_GRASP_ARRAY_LEN
from graspnetAPI.utils.utils import transform_points, parse_posevector
from graspnetAPI.utils.xmlhandler import xmlReader

TOTAL_SCENE_NUM = 25 #190  ori
GRASP_HEIGHT = 0.02

from pyquaternion import Quaternion


# In[5]:


class GraspNetDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='realsense', split='train', num_points=10000,
                 remove_outlier=False, remove_invisible=True, augment=False, load_label=True):
        assert(num_points<=50000)
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
#         self.template_grasp = np.zeros([20000,7])
        if split == 'train':
            self.sceneIds = list( range(5) )
        elif split == 'test':
            self.sceneIds = list( range(100,190) )
        elif split == 'test_seen':
            self.sceneIds = list( range(100,130) )
        elif split == 'test_similar':
            self.sceneIds = list( range(130,160) )
        elif split == 'test_novel':
            self.sceneIds = list( range(160,190) )
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]
        
        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(),  'collision_labels.npz'))
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def augment_data(self, point_clouds, object_poses_list):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            for i in range(len(object_poses_list)):
                object_poses_list[i] = np.dot(flip_mat, object_poses_list[i]).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[1, 0, 0],
                            [0, c,-s],
                            [0, s, c]])
        point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses_list)):
            object_poses_list[i] = np.dot(rot_mat, object_poses_list[i]).astype(np.float32)

        return point_clouds, object_poses_list

    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)
    def loadGrasp(self,index ,format = '6d', camera='realsense', grasp_labels = None, collision_labels = None, fric_coef_thresh=0.4):
        '''
        **Input:**

        - sceneId: int of scene id.

        - annId: int of annotation id.

        - format: string of grasp format, '6d' or 'rect'.

        - camera: string of camera type, 'kinect' or 'realsense'.

        - grasp_labels: dict of grasp labels. Call self.loadGraspLabels if not given.

        - collision_labels: dict of collision labels. Call self.loadCollisionLabels if not given.

        - fric_coef_thresh: float of the frcition coefficient threshold of the grasp. 

        **ATTENTION**

        the LOWER the friction coefficient is, the better the grasp is.

        **Output:**

        - If format == '6d', return a GraspGroup instance.

        - If format == 'rect', return a RectGraspGroup instance.
        '''
        sceneId = int(self.scenename[index][6:])
        annId = self.frameid[index]
        grasp_labels = self.grasp_labels
        collision_labels = self.collision_labels
        import numpy as np
        assert format == '6d' or format == 'rect', 'format must be "6d" or "rect"'
        if format == '6d':
            from graspnetAPI.utils.xmlhandler import xmlReader
            from graspnetAPI.utils.utils import get_obj_pose_list, generate_views, get_model_grasps, transform_points
            from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
            
            camera_poses = np.load(os.path.join(self.root,'scenes','scene_%04d' %(sceneId,),camera, 'camera_poses.npy'))
            camera_pose = camera_poses[annId]
            scene_reader = xmlReader(os.path.join(self.root,'scenes','scene_%04d' %(sceneId,),camera,'annotations','%04d.xml' %(annId,)))
            pose_vectors = scene_reader.getposevectorlist()

            obj_list,pose_list = get_obj_pose_list(camera_pose,pose_vectors)
            if grasp_labels is None:
                print('warning: grasp_labels are not given, calling self.loadGraspLabels to retrieve them')
                grasp_labels = self.loadGraspLabels(objIds = obj_list)
            if collision_labels is None:
                print('warning: collision_labels are not given, calling self.loadCollisionLabels to retrieve them')
                collision_labels = self.loadCollisionLabels(sceneId)

            num_views, num_angles, num_depths = 300, 12, 4
            template_views = generate_views(num_views)
            template_views = template_views[np.newaxis, :, np.newaxis, np.newaxis, :]
            template_views = np.tile(template_views, [1, 1, num_angles, num_depths, 1])

            collision_dump = collision_labels['scene_'+str(sceneId).zfill(4)]

            # grasp = dict()
            grasp_group = GraspGroup()
            for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
                sampled_points, offsets, fric_coefs = grasp_labels[obj_idx+1]
                collision = collision_dump[i]
                point_inds = np.arange(sampled_points.shape[0])

                num_points = len(point_inds)
                target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]
                target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])
                views = np.tile(template_views, [num_points, 1, 1, 1, 1])
                angles = offsets[:, :, :, :, 0]
                depths = offsets[:, :, :, :, 1]
                widths = offsets[:, :, :, :, 2]

                mask1 = ((fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision)
                target_points = target_points[mask1]
                target_points = transform_points(target_points, trans)
                target_points = transform_points(target_points, np.linalg.inv(camera_pose))
                views = views[mask1]
                angles = angles[mask1]
                depths = depths[mask1]
                widths = widths[mask1]
                fric_coefs = fric_coefs[mask1]

                Rs = batch_viewpoint_params_to_matrix(-views, angles)
                Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)
                Rs = np.matmul(np.linalg.inv(camera_pose)[np.newaxis,:3,:3], Rs)

                num_grasp = widths.shape[0]
                scores = (1.1 - fric_coefs).reshape(-1,1)
                widths = widths.reshape(-1,1)
                heights = GRASP_HEIGHT * np.ones((num_grasp,1))
                depths = depths.reshape(-1,1)
                rotations = Rs.reshape((-1,9))
                object_ids = obj_idx * np.ones((num_grasp,1), dtype=np.int32)

                obj_grasp_array = np.hstack([scores, widths, heights, depths, rotations, target_points, object_ids]).astype(np.float32)

                grasp_group.grasp_group_array = np.concatenate((grasp_group.grasp_group_array, obj_grasp_array))
            return grasp_group
        else:
            # 'rect'
            rect_grasps = RectGraspGroup(os.path.join(self.root,'scenes','scene_%04d' % sceneId,camera,'rect','%04d.npy' % annId))
            return rect_grasps

    def get_data(self, index, return_raw_cloud=False):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        try:
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]
        if return_raw_cloud:
            return cloud_masked, color_masked

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['cloud_colors'] = color_sampled.astype(np.float32)

        return ret_dict
    
    def get_data_label(self, index):
        #check npy data without loss
        root_path = '/home/po/TM5/graspnet-baseline/dataset/'
        savegrasp_path = 'pre_grasp_v2'
        savepoints_path = 'pre_point_cloud'
        savecolor_path = 'pre_color'
        data_grasp = 'pre_grasp'
        data_point_clouds = 'pre_point_clouds'
        data_color ='pre_color'
        
       

        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]
        # # save_path = '/home/po/TM5/graspnet-baseline/dataset/pre_grasp'
        # # pre_grasp = np.load('{}/{}_pre_grasp.npy'.format(save_path, index))
        # pre_points = np.load('{}/{}_{}.npy'.format(root_path + savepoints_path, index,data_point_clouds))
        # pre_color = np.load('{}/{}_{}.npy'.format(root_path + savecolor_path, index,data_color))
        # pre_grasp = np.load('{}/{}_{}.npy'.format(root_path + savegrasp_path, index,data_grasp))
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        # generate cloud
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        depth_mask = (depth > 0)
        seg_mask = (seg > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        # sample points
        if len(cloud_masked) >= self.num_points:
            idxs = np.random.choice(len(cloud_masked), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_points-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
#         seg_sampled = seg_masked[idxs]
#         objectness_label = seg_sampled.copy()
#         objectness_label[objectness_label>1] = 1
        
#         object_poses_list = []
#         grasp_points_list = []
#         grasp_offsets_list = []
#         grasp_scores_list = []
#         grasp_tolerance_list = []
#         for i, obj_idx in enumerate(obj_idxs):
#             if obj_idx not in self.valid_obj_idxs:
#                 continue
#             if (seg_sampled == obj_idx).sum() < 50:
#                 continue
#             object_poses_list.append(poses[:, :, i])
#             points, offsets, scores, tolerance = self.grasp_labels[obj_idx]
#             collision = self.collision_labels[scene][i] #(Np, V, A, D)

#             # remove invisible grasp points
#             if self.remove_invisible:
#                 visible_mask = remove_invisible_grasp_points(cloud_sampled[seg_sampled==obj_idx], points, poses[:,:,i], th=0.01)
#                 points = points[visible_mask]
#                 offsets = offsets[visible_mask]
#                 scores = scores[visible_mask]
#                 tolerance = tolerance[visible_mask]
#                 collision = collision[visible_mask]

#             idxs = np.random.choice(len(points), min(max(int(len(points)/4),300),len(points)), replace=False)
#             grasp_points_list.append(points[idxs])
#             grasp_offsets_list.append(offsets[idxs])
#             collision = collision[idxs].copy()
#             scores = scores[idxs].copy()
#             scores[collision] = 0
#             grasp_scores_list.append(scores)
#             tolerance = tolerance[idxs].copy()
#             tolerance[collision] = 0
#             grasp_tolerance_list.append(tolerance)
        
#         if self.augment:
#             cloud_sampled, object_poses_list = self.augment_data(cloud_sampled, object_poses_list)
# ======================================ori tgenerate_tolerance_label.pyte
        
        # grasp_list = self.loadGrasp(index)#.nms()
        grasp_list = self.loadGrasp(index).random_sample( numGrasp = 100)
        # T88 = self.loadGrasp(index)#.nms()
#             -save_path = 'pre_grasp'
#             -pre_grasp = np.load('{}/{}_pre_grasp.npy'.format(save_path, index))
# ======================================
#         for i in T88.__len__():
#             T88[i].


        template_grasp = np.zeros([20000,8])

        count = 0
        label_count = 0
        
        for i in range(grasp_list.__len__()):
            grasp_NOi = grasp_list[i]
            xyz_NOi = grasp_NOi.translation
            q_NOi = grasp_NOi.rotation_matrix
            scorei = grasp_NOi.score
    #         print('grasp_i:',i)
            for j in range(len(cloud_sampled)):
                result = distance_by_translation_point(xyz_NOi,cloud_sampled[j])
    #             print('j:',j)
                if result < 0.0025 :
                    try:
    #                     train_dataset.template_grasp[j,3:7] = Quaternion(matrix=grasp_NOi.rotation_matrix).elements
    #                     train_dataset.template_grasp[j,0:3] = grasp_NOi.translation
                        template_grasp[j,3:7] = Quaternion(matrix=q_NOi).elements
                        template_grasp[j,0:3] = xyz_NOi#grasp_NOi.translation
                        template_grasp[j,7] = scorei
                        label_count = label_count + 1
    #                     print('count:',count,'\n','label_count:',label_count,'\n','i,j:',i,j,'\n\ng')
    #                     print('grasp_label_no:',j,'grasp_7element:',train_dataset.template_grasp[j,:])
                        
                        break
                    except:
    #                     print('fucki',i,'fuckj',j)
                        break
                count = count + 1
        pre_grasp = template_grasp

        xyz_label = pre_grasp[:,0:3]
        q_label = pre_grasp[:,3:7]
        score_label = pre_grasp[:,7:8]
        ret_dict = {}
        ret_dict["point_clouds"] = cloud_sampled.astype(np.float32)
        ret_dict["cloud_colors"] = color_sampled.astype(np.float32)
        # ret_dict["point_clouds"] = pre_points.astype(np.float32)
        # ret_dict["cloud_colors"] = pre_color.astype(np.float32)
#         ret_dict['objectness_label'] = objectness_label.astype(np.int64)
#         ret_dict['object_poses_list'] = object_poses_list
#         ret_dict['grasp_points_list'] = grasp_points_list
#         ret_dict['grasp_offsets_list'] = grasp_offsets_list
#         ret_dict['grasp_labels_list'] = grasp_scores_list
#         ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict["xyz_label"] = xyz_label
        ret_dict["q_label"] = q_label
        ret_dict["score_label"] = score_label

        return ret_dict

def load_grasp_labels(root):
    obj_names = list(range(88))
    valid_obj_idxs = []
    grasp_labels = {}
    for i, obj_name in enumerate(tqdm(obj_names, desc='Loading grasping labels...')):
#         if i == 18: continue
        valid_obj_idxs.append(i + 1) #here align with label png
        label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        tolerance = np.load(os.path.join(BASE_DIR, 'tolerance', '{}_tolerance.npy'.format(str(i).zfill(3))))
        grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
                                label['scores'].astype(np.float32))#, tolerance)

    return valid_obj_idxs, grasp_labels
# def loadGraspLabels(self, objIds=None):
#         '''
#         **Input:**

#         - objIds: int or list of int of the object ids.

#         **Output:**

#         - a dict of grasplabels of each object. 
#         '''
#         # load object-level grasp labels of the given obj ids
#         objIds = self.objIds if objIds is None else objIds
#         assert _isArrayLike(objIds) or isinstance(objIds, int), 'objIds must be an integer or a list/numpy array of integers'
#         objIds = objIds if _isArrayLike(objIds) else [objIds]
#         graspLabels = {}
#         for i in tqdm(objIds, desc='Loading grasping labels...'):
#             file = np.load(os.path.join(self.root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
#             graspLabels[i] = (file['points'].astype(np.float32), file['offsets'].astype(np.float32), file['scores'].astype(np.float32))
#         return graspLabels
def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
    
def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))