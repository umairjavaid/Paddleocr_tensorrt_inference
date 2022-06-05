#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import argparse
import logging as log
import os
import sys
import copy
import cv2
import numpy as np
from operators import NormalizeImage, ToCHWImage, KeepKeys, DetResizeForTest
from sast_postprocess import SASTPostProcess

def draw_text_det_res(dt_boxes, img_path):
    src_im = cv2.imread(img_path)
    for box in dt_boxes:
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
    return src_im

def build_post_process(config, global_config=None):
    support_dict = ['SASTPostProcess']

    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name == "None":
        return
    if global_config is not None:
        config.update(global_config)
    assert module_name in support_dict, Exception(
        'post process only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class

def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops

def order_points_clockwise(pts):
		xSorted = pts[np.argsort(pts[:, 0]), :]

		# grab the left-most and right-most points from the sorted
		# x-roodinate points
		leftMost = xSorted[:2, :]
		rightMost = xSorted[2:, :]

		# now, sort the left-most coordinates according to their
		# y-coordinates so we can grab the top-left and bottom-left
		# points, respectively
		leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
		(tl, bl) = leftMost

		rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
		(tr, br) = rightMost

		rect = np.array([tl, tr, br, bl], dtype="float32")
		return rect

def clip_det_res(points, img_height, img_width):
		for pno in range(points.shape[0]):
			points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
			points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
		return points

def filter_tag_det_res(dt_boxes, image_shape):
		img_height, img_width= image_shape[0:2]
		
		dt_boxes_new = []
		for box in dt_boxes:
			box = order_points_clockwise(box)
			box = clip_det_res(box, img_height, img_width)
			rect_width = int(np.linalg.norm(box[0] - box[1]))
			rect_height = int(np.linalg.norm(box[0] - box[3]))
			if rect_width <= 3 or rect_height <= 3:
				continue
			dt_boxes_new.append(box)
		dt_boxes = np.array(dt_boxes_new)
		return dt_boxes

def draw_text_det_res(dt_boxes, img_path):
	src_im = img_path #cv2.imread(img_path)
	for box in dt_boxes:
		box = np.array(box).astype(np.int32).reshape(-1, 2)
		cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
	return src_im     


def sorted_boxes(dt_boxes):
	"""
	Sort text boxes in order from top to bottom, left to right
	args:
		dt_boxes(array):detected text boxes with shape [4, 2]
	return:
		sorted boxes(array) with shape [4, 2]
	"""
	num_boxes = dt_boxes.shape[0]
	sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
	_boxes = list(sorted_boxes)

	for i in range(num_boxes - 1):
		if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
				(_boxes[i + 1][0][0] < _boxes[i][0][0]):
			tmp = _boxes[i]
			_boxes[i] = _boxes[i + 1]
			_boxes[i + 1] = tmp
	return _boxes

def get_rotate_crop_image(img, points):
	'''
	img_height, img_width = img.shape[0:2]
	left = int(np.min(points[:, 0]))
	right = int(np.max(points[:, 0]))
	top = int(np.min(points[:, 1]))
	bottom = int(np.max(points[:, 1]))
	img_crop = img[top:bottom, left:right, :].copy()
	points[:, 0] = points[:, 0] - left
	points[:, 1] = points[:, 1] - top
	'''
	assert len(points) == 4, "shape of points must be 4*2"
	img_crop_width = int(
		max(
			np.linalg.norm(points[0] - points[1]),
			np.linalg.norm(points[2] - points[3])))
	img_crop_height = int(
		max(
			np.linalg.norm(points[0] - points[3]),
			np.linalg.norm(points[1] - points[2])))
	pts_std = np.float32([[0, 0], [img_crop_width, 0],
						  [img_crop_width, img_crop_height],
						  [0, img_crop_height]])
	M = cv2.getPerspectiveTransform(points, pts_std)
	dst_img = cv2.warpPerspective(
		img,
		M, (img_crop_width, img_crop_height),
		borderMode=cv2.BORDER_REPLICATE,
		flags=cv2.INTER_CUBIC)
	dst_img_height, dst_img_width = dst_img.shape[0:2]
	if dst_img_height * 1.0 / dst_img_width >= 1.5:
		dst_img = np.rot90(dst_img)
	return dst_img

def infer(image):
	pre_process_list = [{
					
					'DetResizeForTest': {
                    'resize_long': 1536
                }
		}, {
			'NormalizeImage': {
				'std': [0.229, 0.224, 0.225],
				'mean': [0.485, 0.456, 0.406],
				'scale': '1./255.',
				'order': 'hwc'
			}
		}, {
			'ToCHWImage': None
		}, {
			'KeepKeys': {
				'keep_keys': ['image', 'shape']
			}
		}]
	
	preprocess_op = create_operators(pre_process_list)
	ori_im = image.copy()
	data = {'image': image}
	data = transform(data,preprocess_op)
	img, shape_list = data
	img = np.expand_dims(img, axis=0)
	shape_list = np.expand_dims(shape_list, axis=0)
	img = img.copy()
###############################   inference #####################################

	outputs = infer(img)
	
################################# post   #############################################
	preds = {}
	preds['f_border'] = outputs[0]
	preds['f_score'] = outputs[1]
	preds['f_tco'] = outputs[2]
	preds['f_tvo'] = outputs[3]
	output_image = ori_im.copy()
	h, w, _ = output_image.shape
	
	postprocess_params = {}
	postprocess_params['name'] = 'SASTPostProcess'
	postprocess_params["score_thresh"] = 0
	postprocess_params["nms_thresh"] = 0
	postprocess_params["sample_pts_num"] = 2
	postprocess_params["expand_scale"] = 1.0
	postprocess_params["shrink_ratio_of_width"] = 0.3
	postprocess_op = build_post_process(postprocess_params)
	post_result = postprocess_op(preds, shape_list)
	dt_boxes = post_result[0]['points']
	print('dt_boxes',dt_boxes)

	print("----------------------------")
	dt_boxes = filter_tag_det_res(dt_boxes, ori_im.shape)
	# print('dt_boxes',dt_boxes.shape)

	dt_boxes = sorted_boxes(dt_boxes)
	# print('dt_boxes',dt_boxes)
	img_crop_list = []
	for bno in range(len(dt_boxes)):
		tmp_box = copy.deepcopy(dt_boxes[bno])
		print(tmp_box)
		img_crop = get_rotate_crop_image(ori_im, tmp_box)
		img_crop_list.append(img_crop)
	print( img_crop_list)
	out=draw_text_det_res(dt_boxes, output_image)
	cv2.imshow('out',out)
	cv2.waitKey(0)
	# for box in dt_boxes:
	#     b = np.array(box).astype(np.int32).reshape(-1, 2)
	#     print('\n',b,'\n')
	#     print('x1:',b[0][0])
	#     print('y1:',b[0][1])
	#     print('x2:',b[2][0])
	#     print('y2:',b[2][1])
		
	return img_crop_list


if __name__ == '__main__':
	img = cv2.imread("m2d1_0.png")
	infer(img)
