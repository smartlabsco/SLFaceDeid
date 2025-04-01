from genericpath import isfile
from heapq import merge
import paddle
import cv2
import numpy as np
import os
import glob
import random
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel

#faceswap parser iamge
class FacedeidParser:
    def get_id_emb(self, id_net, id_img_path):
        id_img = cv2.imread(id_img_path)
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std
        id_emb, id_feature = id_net(id_img)
        id_emb = l2_norm(id_emb)
        return id_emb, id_feature

    def face_align(self, landmarkModel, image_path, merge_result=False, image_size=224):
        if os.path.isfile(image_path):
            img_list = [image_path]
        else:
            img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
        for path in img_list:
            img = cv2.imread(path)
            landmark = landmarkModel.get(img)
            if landmark is not None:
                base_path = path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                aligned_img, back_matrix = align_img(img, landmark, image_size)
                cv2.imwrite(base_path + '_aligned.png', aligned_img)
                if merge_result:
                    np.save(base_path + '_back.npy', back_matrix)
                

    def parsing(self, image, merge_result=True, need_align=True):
        # load model
        paddle.set_device("cpu")
        faceswap_model = FaceSwap(use_gpu=False)
        id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
        id_net.eval()
        weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')
        source_img = 'data/source/test_f/image_1.jpg'
        base_path = source_img.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        id_emb, id_feature = self.get_id_emb(id_net, base_path + '_aligned.png')
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()

        if need_align:
            landmarkModel = LandmarkModel(name='landmarks')
            landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
            self.face_align(landmarkModel, source_img)
            self.face_align(landmarkModel, image, merge_result=True, image_size=224)

        if os.path.isfile(image):
            img_list = [image]
        else:
            img_list = [os.path.join(image, x) for x in os.listdir(image) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
        for img_path in img_list:
            origin_att_img = cv2.imread(img_path)
            base_path = img_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            att_img = cv2.imread(base_path + '_aligned.png')
            att_img = cv2paddle(att_img)

            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)

            if merge_result:
                back_matrix = np.load(base_path + '_back.npy')
                mask = np.transpose(mask[0].numpy(), (1, 2, 0))
                res = dealign(res, origin_att_img, back_matrix, mask)
            cv2.imwrite(base_path + '_result.png', res)

if __name__ == '__main__':
    parser = FacedeidParser()
    parser.parsing('data/target/444.jpg')