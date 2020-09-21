/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */

// main.cpp - blazepose for openvino - modified by Yasunori Shimura   twitter: @yassim0710

// Note: This program works with BlazePose model for OpenVINO from PINTO model zoo 
//       https://github.com/PINTO0309/PINTO_model_zoo

#define _USE_MATH_DEFINES
#include <cmath>

#include <vector>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#define FULL_POSE
#include "blazepose.h"

#define IMAGE_INPUT    (1)
#define VIDEO_INPUT    (2)
#define CAM_INPUT      (3)

#ifndef FULL_POSE
const std::string MODEL_POSE_DET = "../pose_detection/128x128/FP32/pose_detection";
const std::string MODEL_LM_DET   = "../pose_landmark_upper_body/256x256/FP32/pose_landmark_upper_body";
#else
const std::string MODEL_POSE_DET = "../pose_detection_full/128x128/FP16/pose_detection_full";
  //Input Blob(s):
  //  BlobName:input, Shape:[1, 3, 128, 128], Precision:FP32
  //Output Blob(s):
  //  BlobName:StatefulPartitionedCall/functional_1/classificators/concat, Shape:[1, 896, 1], Precision:FP32
  //  BlobName:StatefulPartitionedCall/functional_1/regressors/concat, Shape:[1, 896, 8], Precision:FP32
const std::string MODEL_LM_DET   = "../pose_landmark_full_body/256x256/FP16/pose_landmark_full_body";
  //Input Blob(s):
  //  BlobName:input, Shape:[1, 3, 256, 256], Precision:FP32
  //Output Blob(s):
  //  BlobName:StatefulPartitionedCall/functional_1/output_segmentation/BiasAdd/Add, Shape:[1, 1, 128, 128], Precision:FP32
  //  BlobName:StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid/Sigmoid, Shape:[1, 1, 1, 1], Precision:FP32
  //  BlobName:StatefulPartitionedCall/functional_1/tf_op_layer_ld_3d/ld_3d, Shape:[1, 156], Precision:FP32
#endif

// INPUT_TYPE = { IMAGE_INPUT | VIDEO_INPUT | CAM_INPUT }
#define INPUT_TYPE    VIDEO_INPUT
const std::string INPUT_FILE = "../boy.mp4";                    /* Image or movie file */
//const std::string INPUT_FILE = "../capoeira.mp4";             /* Image or movie file */
//const std::string INPUT_FILE = "../people-detection.mp4";     /* Image or movie file */

// 'output.mp4' will be generated when this macro is defined and the input source is either one of VIDEO_INPUT or CAM_INPUT
#define VIDEO_OUTPUT
#define VIDEO_SIZE    (400)             /* output video size = (VIDEO_SIZE, VIDEO_SIZE) */

// Device to use for inferencing. Possible options = "CPU", "GPU", "MYRIAD", "HDDL", "HETERO:FPGA,CPU", ...
const std::string DEVICE_PD = "CPU";
const std::string DEVICE_LM = "CPU";

// ** Define or Undefine to control the items to display
//#define RENDER_ROI
#define RENDER_TIME
#define RENDER_POINTS



namespace ie = InferenceEngine;


float CalculateScale(float min_scale, float max_scale, int stride_index, int num_strides) {
    if (num_strides == 1)
        return (min_scale + max_scale) * 0.5f;
    else
        return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
}

void GenerateAnchors(std::vector<Anchor>& anchors, const SsdAnchorsCalculatorOptions& options) {
    int layer_id = 0;
    while (layer_id < (int)options.strides.size()) {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;
        
        // For same strides, we merge the anchors in the same order.
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < (int)options.strides.size() &&
            options.strides[last_same_stride_layer] == options.strides[layer_id])
        {
            const float scale =
                CalculateScale(options.min_scale, options.max_scale,
                    last_same_stride_layer, options.strides.size());
            if (last_same_stride_layer == 0 && options.reduce_boxes_in_lowest_layer) {
                // For first layer, it can be specified to use predefined anchors.
                aspect_ratios.push_back(1.0);
                aspect_ratios.push_back(2.0);
                aspect_ratios.push_back(0.5);
                scales.push_back(0.1);
                scales.push_back(scale);
                scales.push_back(scale);
            }
            else {
                for (int aspect_ratio_id = 0;
                    aspect_ratio_id < (int)options.aspect_ratios.size();
                    ++aspect_ratio_id) {
                    aspect_ratios.push_back(options.aspect_ratios[aspect_ratio_id]);
                    scales.push_back(scale);
                }
                if (options.interpolated_scale_aspect_ratio > 0.0) {
                    const float scale_next =
                        last_same_stride_layer == (int)options.strides.size() - 1
                        ? 1.0f
                        : CalculateScale(options.min_scale, options.max_scale,
                            last_same_stride_layer + 1,
                            options.strides.size());
                    scales.push_back(std::sqrt(scale * scale_next));
                    aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer++;
        }

        for (int i = 0; i < (int)aspect_ratios.size(); ++i) {
            const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        int feature_map_height = 0;
        int feature_map_width = 0;
        if (options.feature_map_height.size()) {
            feature_map_height = options.feature_map_height[layer_id];
            feature_map_width = options.feature_map_width[layer_id];
        }
        else {
            const int stride = options.strides[layer_id];
            feature_map_height = std::ceil(1.0f * options.input_size_height / stride);
            feature_map_width = std::ceil(1.0f * options.input_size_width / stride);
        }

        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (int anchor_id = 0; anchor_id < (int)anchor_height.size(); ++anchor_id) {
                    // TODO: Support specifying anchor_offset_x, anchor_offset_y.
                    const float x_center = (x + options.anchor_offset_x) * 1.0f / feature_map_width;
                    const float y_center = (y + options.anchor_offset_y) * 1.0f / feature_map_height;

                    Anchor new_anchor;
                    new_anchor.x_center = x_center;
                    new_anchor.y_center = y_center;

                    if (options.fixed_anchor_size) {
                        new_anchor.w = 1.0f;
                        new_anchor.h = 1.0f;
                    }
                    else {
                        new_anchor.w = anchor_width[anchor_id];
                        new_anchor.h = anchor_height[anchor_id];
                    }
                    anchors.push_back(new_anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
}

void create_ssd_anchors(int input_w, int input_h, std::vector<Anchor> &anchors) {
    /*
     *  Anchor parameters are based on:
     *      mediapipe/modules/pose_detection/pose_detection_cpu.pbtxt
     */
    SsdAnchorsCalculatorOptions anchor_options;
    anchor_options.num_layers        = 4;
    anchor_options.min_scale         = 0.1484375;
    anchor_options.max_scale         = 0.75;
    anchor_options.input_size_height = 128;
    anchor_options.input_size_width  = 128;
    anchor_options.anchor_offset_x   = 0.5f;
    anchor_options.anchor_offset_y   = 0.5f;
    //  anchor_options.feature_map_width .push_back(0);
    //  anchor_options.feature_map_height.push_back(0);
    anchor_options.strides.push_back(8);
    anchor_options.strides.push_back(16);
    anchor_options.strides.push_back(16);
    anchor_options.strides.push_back(16);
    anchor_options.aspect_ratios.push_back(1.0);
    anchor_options.reduce_boxes_in_lowest_layer = false;
    anchor_options.interpolated_scale_aspect_ratio = 1.0;
    anchor_options.fixed_anchor_size = true;

    GenerateAnchors(anchors, anchor_options);
}

int get_bbox_idx(int anchor_idx) {
    /*
     *  cx, cy, width, height
     *  key0_x, key0_y      kMidHipCenter
     *  key1_x, key1_y      kFullBodySizeRot
     *  key2_x, key2_y      kMidShoulderCenter - upper body only
     *  key3_x, key3_y      kUpperBodySizeRot  - upper body only
     */
    int numkey = kPoseDetectKeyNum;   // FullBody:4, UpperBody:2
    int idx = (4 + 2 * numkey) * anchor_idx;

    return idx;
}

int decode_bounds(std::list<detect_region_t>& region_list, float score_thresh, int input_img_w, int input_img_h, float* scores_ptr, float* bboxes_ptr, std::vector<Anchor>& anchors) {
    detect_region_t region;
    int i = 0;
    for (auto &anchor : anchors) {
        float score0 = scores_ptr[i];
        float score = 1.0f / (1.0f + exp(-score0));

        if (score > score_thresh)
        {
            float* p = bboxes_ptr + get_bbox_idx(i);

            /* boundary box */
            float cx = p[0] / input_img_w + anchor.x_center;
            float cy = p[1] / input_img_h + anchor.y_center;
            float w  = p[2] / input_img_w;
            float h  = p[3] / input_img_h;

            fvec2 topleft, btmright;
            topleft.x  = cx - w * 0.5f;
            topleft.y  = cy - h * 0.5f;
            btmright.x = cx + w * 0.5f;
            btmright.y = cy + h * 0.5f;

            region.score    = score;
            region.topleft  = topleft;
            region.btmright = btmright;

            /* landmark positions (6 keys) */
            for (int j = 0; j < kPoseDetectKeyNum; j++)
            {
                float lx = p[4 + (2 * j) + 0];
                float ly = p[4 + (2 * j) + 1];
                lx += anchor.x_center * input_img_w;
                ly += anchor.y_center * input_img_h;
                lx /= (float)input_img_w;
                ly /= (float)input_img_h;

                region.keys[j].x = lx;
                region.keys[j].y = ly;
            }

            region_list.push_back(region);
        }
        i++;
    }
    return 0;
}

/* -------------------------------------------------- *
 *  Apply NonMaxSuppression:
 *      https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/image_ops.ts
 * -------------------------------------------------- */
float calc_intersection_over_union(detect_region_t& region0, detect_region_t& region1) {
    float sx0 = region0.topleft.x;
    float sy0 = region0.topleft.y;
    float ex0 = region0.btmright.x;
    float ey0 = region0.btmright.y;
    float sx1 = region1.topleft.x;
    float sy1 = region1.topleft.y;
    float ex1 = region1.btmright.x;
    float ey1 = region1.btmright.y;

    float xmin0 = std::min(sx0, ex0);
    float ymin0 = std::min(sy0, ey0);
    float xmax0 = std::max(sx0, ex0);
    float ymax0 = std::max(sy0, ey0);
    float xmin1 = std::min(sx1, ex1);
    float ymin1 = std::min(sy1, ey1);
    float xmax1 = std::max(sx1, ex1);
    float ymax1 = std::max(sy1, ey1);

    float area0 = (ymax0 - ymin0) * (xmax0 - xmin0);
    float area1 = (ymax1 - ymin1) * (xmax1 - xmin1);
    if (area0 <= 0 || area1 <= 0)
        return 0.0f;

    float intersect_xmin = std::max(xmin0, xmin1);
    float intersect_ymin = std::max(ymin0, ymin1);
    float intersect_xmax = std::min(xmax0, xmax1);
    float intersect_ymax = std::min(ymax0, ymax1);

    float intersect_area = std::max(intersect_ymax - intersect_ymin, 0.0f) *
        std::max(intersect_xmax - intersect_xmin, 0.0f);

    return intersect_area / (area0 + area1 - intersect_area);
}


int non_max_suppression(std::list<detect_region_t>& region_list, std::list<detect_region_t>& region_nms_list, float iou_thresh) {
    region_list.sort([](detect_region_t& v1, detect_region_t& v2) { return v1.score > v2.score ? true : false; });

    for (auto itr = region_list.begin(); itr != region_list.end(); itr++)
    {
        detect_region_t region_candidate = *itr;

        int ignore_candidate = false;
        for (auto itr_nms = region_nms_list.rbegin(); itr_nms != region_nms_list.rend(); itr_nms++)
        {
            detect_region_t region_nms = *itr_nms;

            float iou = calc_intersection_over_union(region_candidate, region_nms);
            if (iou >= iou_thresh)
            {
                ignore_candidate = true;
                break;
            }
        }

        if (!ignore_candidate)
        {
            region_nms_list.push_back(region_candidate);
            if (region_nms_list.size() >= MAX_POSE_NUM)
                break;
        }
    }
    return 0;
}

float normalize_radians(float angle)
{
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

void compute_rotation(detect_region_t& region) {
#ifndef FULL_POSE
    float x0 = region.keys[kMidHipCenter].x;
    float y0 = region.keys[kMidHipCenter].y;
    float x1 = region.keys[kMidShoulderCenter].x;
    float y1 = region.keys[kMidShoulderCenter].y;
#else
    float x0 = region.keys[kMidHipCenter].x;
    float y0 = region.keys[kMidHipCenter].y;
    float x1 = (region.topleft.x + region.btmright.x) * 0.5f;
    float y1 = (region.topleft.y + region.btmright.y) * 0.5f;
#endif
    float target_angle = M_PI * 0.5f;
    float rotation = target_angle - std::atan2(-(y1 - y0), x1 - x0);

    region.rotation = normalize_radians(rotation);
}

void rot_vec(fvec2& vec, float rotation) {
    float sx = vec.x;
    float sy = vec.y;
    vec.x = sx * std::cos(rotation) - sy * std::sin(rotation);
    vec.y = sx * std::sin(rotation) + sy * std::cos(rotation);
}

void compute_detect_to_roi(detect_region_t& region, const ie::SizeVector& dims) {
    int input_img_w = dims[3];
    int input_img_h = dims[2];
#ifndef FULL_POSE
    float x_center = region.keys[kMidShoulderCenter].x * input_img_w;
    float y_center = region.keys[kMidShoulderCenter].y * input_img_h;
    float x_scale  = region.keys[kUpperBodySizeRot ].x * input_img_w;
    float y_scale  = region.keys[kUpperBodySizeRot ].y * input_img_h;
#else
    float x_center = region.keys[kMidHipCenter   ].x * input_img_w;
    float y_center = region.keys[kMidHipCenter   ].y * input_img_h;
    float x_scale  = region.keys[kFullBodySizeRot].x * input_img_w;
    float y_scale  = region.keys[kFullBodySizeRot].y * input_img_h;
#endif
    // Bounding box size as double distance from center to scale point.
    float box_size = std::sqrt((x_scale - x_center) * (x_scale - x_center) +
                               (y_scale - y_center) * (y_scale - y_center)) * 2.0;

    /* RectTransformationCalculator::TransformNormalizedRect() */
    float width = box_size;
    float height = box_size;
    float rotation = region.rotation;
    float shift_x = 0.0f;
    float shift_y = 0.0f;
    float roi_cx;
    float roi_cy;

    if (rotation == 0.0f) {
        roi_cx = x_center + (width  * shift_x);
        roi_cy = y_center + (height * shift_y);
    } else {
        float dx = (width * shift_x) * std::cos(rotation) - (height * shift_y) * std::sin(rotation);
        float dy = (width * shift_x) * std::sin(rotation) + (height * shift_y) * std::cos(rotation);
        roi_cx = x_center + dx;
        roi_cy = y_center + dy;
    }

    /*
     *  calculate ROI width and height.
     *  scale parameter is based on
     *      "mediapipe/modules/pose_landmark/pose_detection_to_roi.pbtxt"
     */
    float scale_x = 1.5f;
    float scale_y = 1.5f;
    float long_side = std::max(width, height);
    float roi_w = long_side * scale_x;
    float roi_h = long_side * scale_y;

    region.roi_center.x = roi_cx / (float)input_img_w;
    region.roi_center.y = roi_cy / (float)input_img_h;
    region.roi_size.x = roi_w / (float)input_img_w;
    region.roi_size.y = roi_h / (float)input_img_h;

    /* calculate ROI coordinates */
    float dx = roi_w * 0.5f;
    float dy = roi_h * 0.5f;
    region.roi_coord[0].x = -dx;  region.roi_coord[0].y = -dy;
    region.roi_coord[1].x = +dx;  region.roi_coord[1].y = -dy;
    region.roi_coord[2].x = +dx;  region.roi_coord[2].y = +dy;
    region.roi_coord[3].x = -dx;  region.roi_coord[3].y = +dy;

    for (int i = 0; i < 4; i++)
    {
        rot_vec(region.roi_coord[i], rotation);
        region.roi_coord[i].x += roi_cx;
        region.roi_coord[i].y += roi_cy;

        region.roi_coord[i].x /= (float)input_img_h;
        region.roi_coord[i].y /= (float)input_img_h;
    }
}


static void pack_detect_result(std::vector<detect_region_t>&detect_results, std::list<detect_region_t>& region_list, const ie::SizeVector& dims) {
    for (auto& region : region_list) {
        compute_rotation(region);
        compute_detect_to_roi(region, dims);
        detect_results.push_back(region);
    }
}


void dumpPose(detect_region_t& pose) {
    std::cout << "Score :" << pose.score << ", topleft: (" << pose.topleft.x << ", " << pose.topleft.y << ") btmright(" << pose.btmright.x << ", " << pose.btmright.y << ")" << std::endl;
    std::cout << "keys: ";
    for (size_t i = 0; i < kPoseDetectKeyNum; i++) {
        std::cout << "(" << pose.keys[i].x << ", " << pose.keys[i].y << ") ";
    }
    std::cout << std::endl;
    std::cout << "rotation(" << pose.rotation << ", roi_center(" << pose.roi_center.x << ", " << pose.roi_center.y << "), roi_size(" << pose.roi_size.x << ", " << pose.roi_size.y << ")" << std::endl;
    for (size_t i = 0; i < 4; i++) {
        std::cout << "roi_coord[" << i << "]=(" << pose.roi_coord[i].x << ", " << pose.roi_coord[i].y << ") ";
    }
    std::cout << std::endl;
}


void renderROI(cv::Mat& img, const fvec2(&roi_coord)[4]) {
    size_t w = img.size().width;
    size_t h = img.size().height;
    std::vector<cv::Point> p;
    for (size_t i = 0; i < 4; i++) {
        cv::Point pt = cv::Point(roi_coord[i].x * w, roi_coord[i].y * h);
        p.push_back(pt);
        cv::putText(img, std::to_string(i), pt, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1);
    }
    cv::polylines(img, p, true, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
}



void renderPose(cv::Mat& image_ocv, std::vector<detect_region_t>& detect_results, pose_landmark_result_t *landmarks) {
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255,   0,   0), cv::Scalar(255,  85,   0), cv::Scalar(255, 170,   0),
        cv::Scalar(255, 255,   0), cv::Scalar(170, 255,   0), cv::Scalar( 85, 255,   0),
        cv::Scalar(  0, 255,   0), cv::Scalar(  0, 255,  85), cv::Scalar(  0, 255, 170),
        cv::Scalar(  0, 255, 255), cv::Scalar(  0, 170, 255), cv::Scalar(  0,  85, 255),
        cv::Scalar(  0,   0, 255), cv::Scalar( 85,   0, 255), cv::Scalar(170,   0, 255),
        cv::Scalar(255,   0, 255), cv::Scalar(255,   0, 170), cv::Scalar(255,   0,  85)
    };
    const std::vector<std::vector<int>> bones = {
        {  0,  1,  2,  3,  7},          // 0 right eye and ear
        {  0,  4,  5,  6,  8},          // 1 left eye and ear
        {  9, 10},                      // 2 mouth
        { 11, 12, 24, 23, 11},          // 3 body
        { 11, 13, 15, 17, 19, 15, 21},  // 4 right arm
        { 12, 14, 16, 18, 20, 16, 22}   // 5 left arm
#ifdef FULL_POSE
       ,{ 23, 25, 27, 29, 31, 27 },     // 6 right leg
        { 24, 26, 28, 30, 32, 28}       // 7 left leg
#endif
    };
    size_t W = image_ocv.size().width;
    size_t H = image_ocv.size().height;
    for (size_t pose_id = 0; pose_id < detect_results.size(); pose_id++) {
        detect_region_t pose = detect_results[pose_id];
        cv::Point2f mat_src[] = { cv::Point2f(0.f, 0.f), cv::Point2f(1.f, 0.f), cv::Point2f(1.f, 1.f) };
        cv::Point2f mat_dst[] = { cv::Point2f(pose.roi_coord[0].x * W, pose.roi_coord[0].y * H),
                                  cv::Point2f(pose.roi_coord[1].x * W, pose.roi_coord[1].y * H),
                                  cv::Point2f(pose.roi_coord[2].x * W, pose.roi_coord[2].y * H) };
        cv::Mat mat = cv::getAffineTransform(mat_src, mat_dst);
        mat.resize(3, cv::Scalar(0.f));
        mat.at<double>(2, 2) = 1.f;

        // Apply affine transform to project the junction points to the original ROI in the input image
        double px, py, flag;
        std::vector<std::pair<cv::Point2f, float>> pts;
        for (size_t i = 0; i < POSE_JOINT_NUM; i++) {
            cv::Mat pt = (cv::Mat_<double>(3, 1) << landmarks[pose_id].joint[i].x, landmarks[pose_id].joint[i].y, 1.f);   // 1x3 matrix
            pt = mat * pt;                                              // Affine transform (0,0)-(1,1)  => ROI
            px = pt.at<double>(0, 0);
            py = pt.at<double>(1, 0);
            flag = landmarks[pose_id].joint[i].z;
            pts.push_back(std::pair<cv::Point2f, float>(cv::Point2f(px, py), flag));
            cv::circle(image_ocv, cv::Point(pt.at<double>(0, 0), pt.at<double>(1, 0)), 4, cv::Scalar(255, 0, 0), -1);
        }
        // render bones
        size_t bone_idx = 0;
        for (auto& bone : bones) {
            size_t prev_idx = -1;
            for (auto& idx : bone) {
                if (prev_idx == -1) {
                    prev_idx = idx;
                    continue;
                }
                cv::Point2f pt1 = pts[prev_idx].first;
                cv::Point2f pt2 = pts[     idx].first;
                float flag1 = pts[prev_idx].second;
                float flag2 = pts[     idx].second;
                cv::Scalar color = colors[bone_idx++ % colors.size()];
                cv::line(image_ocv, pt1, pt2, color, 2);
                prev_idx = idx;
            }
        }
#ifdef RENDER_POINTS
        for (auto& pt : pts ) {
            cv::circle(image_ocv, pt.first, 4, cv::Scalar(255, 0, 0), -1);
        }
#endif
    }
}

void renderTime(cv::Mat& img, double time_pd, double time_lm, double time_ttl) {

    cv::putText(img, cv::format("PD: %6.2fms", time_pd/1000.f), cv::Point(0, 10), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
    cv::putText(img, cv::format("LM: %6.2fms", time_lm/1000.f), cv::Point(0, 30), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
    cv::putText(img, cv::format("TTL: %6.2fms", time_ttl / 1000.f), cv::Point(0, 50), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 1);
}



int main(int argc, char *argv[]) {

    std::chrono::system_clock::time_point start, end, start_ttl, end_ttl;
    std::chrono::microseconds time_pd, time_lm, time_ttl;

    ie::Core ie;

    // Initialization for pose model
    ie::CNNNetwork net_pd;
    net_pd = ie.ReadNetwork(MODEL_POSE_DET+".xml", MODEL_POSE_DET+".bin");
    net_pd.setBatchSize(1);
    std::string input_name_pd = "input";                                                               // [1,3,128,128]
    std::shared_ptr<ie::InputInfo> input_info_pd = net_pd.getInputsInfo()[input_name_pd];
    const ie::SizeVector idims_pd = input_info_pd->getTensorDesc().getDims();                          // 0,1,2,3 = N,C,H,W
    input_info_pd->setLayout(ie::Layout::NHWC);
    input_info_pd->setPrecision(ie::Precision::FP32);

    std::vector<std::string> oname_pd = { "StatefulPartitionedCall/functional_1/classificators/concat" ,    // [1,896,1]    tensor scores
                                          "StatefulPartitionedCall/functional_1/regressors/concat" };       // [1,896,12]   tensor bboxes
    for (auto& oname : oname_pd) {
        ie::DataPtr output_info = net_pd.getOutputsInfo()[oname];
        output_info->setPrecision(ie::Precision::FP32);
    }

    ie::ExecutableNetwork exenet_pd = ie.LoadNetwork(net_pd, DEVICE_PD);
    ie::InferRequest ireq_pd = exenet_pd.CreateInferRequest();



    // Initialization for landmark model
    ie::CNNNetwork net_lm;
    net_lm = ie.ReadNetwork(MODEL_LM_DET + ".xml", MODEL_LM_DET + ".bin");
    net_lm.setBatchSize(1);
    std::string input_name_lm = "input";                                                                // [1,3,256,256]
    std::shared_ptr<ie::InputInfo> input_info_lm = net_lm.getInputsInfo()[input_name_lm];
    const ie::SizeVector idims_lm = input_info_lm ->getTensorDesc().getDims();                          // 0,1,2,3 = N,C,H,W
    input_info_lm->setLayout(ie::Layout::NHWC);
    input_info_lm->setPrecision(ie::Precision::FP32);

    std::vector<std::string> oname_lm = { "StatefulPartitionedCall/functional_1/output_segmentation/BiasAdd/Add",   // [1, 1, 128, 128]
                                          "StatefulPartitionedCall/functional_1/tf_op_layer_Sigmoid/Sigmoid",       // [1, 1, 1, 1]      landmark flag ?
                                          "StatefulPartitionedCall/functional_1/tf_op_layer_ld_3d/ld_3d" };         // [1, 124]          landmark
    for (auto& oname : oname_lm) {
        ie::DataPtr output_info = net_lm.getOutputsInfo()[oname];
        output_info->setPrecision(ie::Precision::FP32);
    }

    ie::ExecutableNetwork exenet_lm = ie.LoadNetwork(net_lm, DEVICE_LM);
    ie::InferRequest ireq_lm = exenet_lm.CreateInferRequest();



    std::vector<Anchor> anchors;
    create_ssd_anchors(idims_pd[3], idims_pd[2], anchors);

#if   INPUT_TYPE == VIDEO_INPUT
    cv::VideoCapture cap(INPUT_FILE);
#elif INPUT_TYPE == CAM_INPUT
    cv::VideoCapture cap(0);
#endif

#if INPUT_TYPE == VIDEO_INPUT || INPUT_TYPE == CAM_INPUT
#ifdef VIDEO_OUTPUT
    cv::VideoWriter writer("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, cv::Size(VIDEO_SIZE, VIDEO_SIZE));
#endif
#endif


    int key = -1;
    while (key != 27) {     // ESC key to quit

        // read an image and trim down to square image
        const size_t _N = 0, _C = 1, _H = 2, _W = 3;
        cv::Mat image_ocv, image_pd, image_lm, image_f;
        cv::Mat image_org;
#if INPUT_TYPE == IMAGE_INPUT
        image_org = cv::imread(INPUT_FILE);
#elif INPUT_TYPE == VIDEO_INPUT || INPUT_TYPE == CAM_INPUT
        bool flg = cap.read(image_org);
        if (flg == false) break;                        // end of the movie file
#endif
        start_ttl = std::chrono::system_clock::now();

        // resize input image with keeping aspect ratio
        size_t iw = image_org.size().width;
        size_t ih = image_org.size().height;
        if (iw > ih) {
            image_ocv = image_org(cv::Rect(iw / 2 - ih / 2, 0, ih, ih));
        }
        else if (iw < ih) {
            image_ocv = image_org(cv::Rect(0, ih / 2 - iw / 2, iw, iw));
        }

        // pose (ROI) detection
        cv::resize(image_ocv, image_pd, cv::Size(idims_pd[_W], idims_pd[_H]));
        //image_pd.convertTo(image_f, CV_32F, (1.f / 255.f), 0.f);                    // Convert to FP32 and do pre-processes (scale and mean subtract)
        image_pd.convertTo(image_f, CV_32F, (1.f / 127.5f), -1.f);                    // Convert to FP32 and do pre-processes (scale and mean subtract)
        ie::TensorDesc tDesc(ie::Precision::FP32, { 1, static_cast<unsigned int>(image_f.channels()), static_cast<unsigned int>(image_f.size().height), static_cast<unsigned int>(image_f.size().width) }, ie::Layout::NHWC);
        ireq_pd.SetBlob(input_name_pd, ie::make_shared_blob<float>(tDesc, (float*)(image_f.data)));

        start = std::chrono::system_clock::now();
        ireq_pd.Infer();                                                            // Pose detection
        end = std::chrono::system_clock::now();
        time_pd = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        //  BlobName:StatefulPartitionedCall/functional_1/classificators/concat, Shape:[1, 896, 1], Precision:FP32
        //  BlobName:StatefulPartitionedCall/functional_1/regressors/concat, Shape:[1, 896, 8], Precision:FP32
        float* scores = ireq_pd.GetBlob(oname_pd[0])->buffer();
        float* bboxes = ireq_pd.GetBlob(oname_pd[1])->buffer();

        std::list<detect_region_t> region_list, region_nms_list;
        std::vector<detect_region_t> detect_results;
        decode_bounds(region_list, 0.5f, idims_pd[_W], idims_pd[_H], scores, bboxes, anchors);
        non_max_suppression(region_list, region_nms_list, 0.3f);
        pack_detect_result(detect_results, region_nms_list, idims_pd);

        // landmark detection
        cv::Mat outimg = image_ocv.clone();             // clone input image for rendering
        size_t W = image_ocv.size().width;
        size_t H = image_ocv.size().height;
        pose_landmark_result_t  landmarks[MAX_POSE_NUM] = { 0 };
        for (size_t pose_id = 0; pose_id < detect_results.size(); pose_id++) {

            detect_region_t pose = detect_results[pose_id];

            cv::Point2f mat_src[] = { cv::Point2f(pose.roi_coord[0].x * W, pose.roi_coord[0].y * H),
                                      cv::Point2f(pose.roi_coord[1].x * W, pose.roi_coord[1].y * H),
                                      cv::Point2f(pose.roi_coord[2].x * W, pose.roi_coord[2].y * H) };
            cv::Point2f mat_dst[] = { cv::Point2f(0, 0), cv::Point2f(idims_lm[_W], 0), cv::Point2f(idims_lm[_W], idims_lm[_H]) };
            cv::Mat mat = cv::getAffineTransform(mat_src, mat_dst);
            cv::Mat img_affine = cv::Mat::zeros(idims_lm[_W], idims_lm[_H], CV_8UC3);
            cv::warpAffine(image_ocv, img_affine, mat, img_affine.size());              // Crop and rotate ROI by warp affine transform

            //img_affine.convertTo(image_f, CV_32F, (1.f / 255.f), 0.f);
            img_affine.convertTo(image_f, CV_32F, (1.f / 127.5f), -1.f);
            tDesc = ie::TensorDesc(ie::Precision::FP32, { 1, (unsigned int)(image_f.channels()), (unsigned int)(image_f.size().height), (unsigned int)(image_f.size().width) }, ie::Layout::NHWC);
            ireq_lm.SetBlob(input_name_lm, ie::make_shared_blob<float>(tDesc, (float*)(image_f.data)));

            start = std::chrono::system_clock::now();
            ireq_lm.Infer();                                                // Landmark detection
            end = std::chrono::system_clock::now();
            time_lm = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            float* output_lm1 = ireq_lm.GetBlob(oname_lm[0])->buffer();     // ??
            float* lm_score   = ireq_lm.GetBlob(oname_lm[1])->buffer();     // Landmark score (flag)
            float* lm         = ireq_lm.GetBlob(oname_lm[2])->buffer();     // Landmarks

            landmarks[pose_id].score = *lm_score;
            for (size_t i = 0; i < POSE_JOINT_NUM; i++) {
                landmarks[pose_id].joint[i].x = lm[4 * i + 0] / (float)idims_lm[_W];
                landmarks[pose_id].joint[i].y = lm[4 * i + 1] / (float)idims_lm[_H];
                landmarks[pose_id].joint[i].z = lm[4 * i + 2];
            }
#ifdef RENDER_ROI
            renderROI(outimg, pose.roi_coord);
#endif
            renderPose(outimg, detect_results, landmarks);
        }
        end_ttl = std::chrono::system_clock::now();
        time_ttl = std::chrono::duration_cast<std::chrono::microseconds>(end_ttl - start_ttl);

#ifdef RENDER_TIME
        renderTime(outimg, time_pd.count(), time_lm.count(), time_ttl.count());
#endif

#if INPUT_TYPE == IMAGE_INPUT
        cv::imwrite("output.jpg", outimg);
        cv::imshow("output", outimg);
        cv::waitKey(0);
        key = 27;       // force exit
#elif INPUT_TYPE == VIDEO_INPUT || INPUT_TYPE == CAM_INPUT
        cv::imshow("output", outimg);
        key = cv::waitKey(1);
#ifdef VIDEO_OUTPUT
        cv::resize(outimg, outimg, cv::Size(VIDEO_SIZE, VIDEO_SIZE));
        writer << outimg;
#endif
#endif
    }
    return 0;
}
