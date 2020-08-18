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

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

#include "blazepose.h"

const std::string MODEL_POSE_DET = "../pose_detection/128x128/FP16/pose_detection";
const std::string MODEL_LM_DET   = "../pose_landmark_upper_body/256x256/FP16/pose_landmark_upper_body";
const std::string INPUT_IMAGE    = "../test5.jpg";

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

float* get_bbox_ptr(int anchor_idx, float* bboxes_ptr) {
    /*
     *  cx, cy, width, height
     *  key0_x, key0_y
     *  key1_x, key1_y
     *  key2_x, key2_y
     *  key3_x, key3_y
     */
    int numkey = kPoseDetectKeyNum;
    int idx = (4 + 2 * numkey) * anchor_idx;

    return &(bboxes_ptr[idx]);
}

int decode_bounds(std::list<detect_region_t>& region_list, float score_thresh, int input_img_w, int input_img_h, float* scores_ptr, float* bboxes_ptr, std::vector<Anchor>& anchors) {
    detect_region_t region;
    int i = 0;
    for (auto &anchor : anchors) {
        float score0 = scores_ptr[i];
        float score = 1.0f / (1.0f + exp(-score0));
        //std::cout << score << std::endl;

        if (score > score_thresh)
        {
            float* p = get_bbox_ptr(i, bboxes_ptr);

            /* boundary box */
            float sx = p[0];
            float sy = p[1];
            float w  = p[2];
            float h  = p[3];

            float cx = sx + anchor.x_center * input_img_w;
            float cy = sy + anchor.y_center * input_img_h;

            cx /= (float)input_img_w;
            cy /= (float)input_img_h;
            w  /= (float)input_img_w;
            h  /= (float)input_img_h;

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
    float x0 = region.keys[kMidHipCenter].x;
    float y0 = region.keys[kMidHipCenter].y;
    float x1 = region.keys[kMidShoulderCenter].x;
    float y1 = region.keys[kMidShoulderCenter].y;

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
    float x_center = region.keys[kMidShoulderCenter].x * input_img_w;
    float y_center = region.keys[kMidShoulderCenter].y * input_img_h;
    float x_scale  = region.keys[kUpperBodySizeRot] .x * input_img_w;
    float y_scale  = region.keys[kUpperBodySizeRot] .y * input_img_h;

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
        p.push_back(cv::Point(roi_coord[i].x * w, roi_coord[i].y * h));
    }
    cv::polylines(img, p, true, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
}



void renderPose(cv::Mat& image_ocv, std::vector<detect_region_t>& detect_results, pose_landmark_result_t *landmarks) {
    const std::vector<std::vector<int>> bones = {
        {  0,  1,  2,  3,  7},
        {  0,  4,  5,  6,  8},
        {  9, 10},
        { 11, 12, 24, 23, 11},
        { 11, 13, 15, 17, 19, 15, 21},
        { 12, 14, 16, 18, 20, 16, 22}
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

        // Apply affine transform to the junction points
        double px, py, flag;
        std::vector<std::pair<cv::Point2f, float>> pts;
        for (size_t i = 0; i < POSE_JOINT_NUM; i++) {
            cv::Mat pt = (cv::Mat_<double>(3, 1) << landmarks[pose_id].joint[i].x, landmarks[pose_id].joint[i].y, 1.f);   // 1x3 matrix
            pt = mat * pt;                                              // Affine transform (0,0)-(1,1)  => ROI
            px = pt.at<double>(0, 0);
            py = pt.at<double>(1, 0);
            flag = landmarks[pose_id].joint[i].z;
            pts.push_back(std::pair<cv::Point2f, float>(cv::Point2f(px, py), flag));
            //cv::circle(image_ocv, cv::Point(pt.at<double>(0, 0), pt.at<double>(1, 0)), 4, cv::Scalar(255, 0, 0), -1);
        }
        // render bones
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
                cv::line(image_ocv, pt1, pt2, cv::Scalar(255, 0, 255), 2);
                prev_idx = idx;
            }
        }
    }
}


int main(int argc, char *argv[]) {

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

    ie::ExecutableNetwork exenet_pd = ie.LoadNetwork(net_pd, "CPU");
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

    ie::ExecutableNetwork exenet_lm = ie.LoadNetwork(net_lm, "CPU");
    ie::InferRequest ireq_lm = exenet_lm.CreateInferRequest();



    std::vector<Anchor> anchors;
    create_ssd_anchors(idims_pd[3], idims_pd[2], anchors);


    // read an image and trim down to square image
    const size_t _N = 0, _C = 1, _H = 2, _W = 3;
    cv::Mat image_ocv, image_pd, image_lm, image_f;
    cv::Mat image_org = cv::imread(INPUT_IMAGE);
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
    image_pd.convertTo(image_f, CV_32F, (1/127.5f), -1.f);                      // Convert to FP32 and do pre-processes (scale and mean subtract)
    ie::TensorDesc tDesc(ie::Precision::FP32, {1, static_cast<unsigned int>(image_f.channels()), static_cast<unsigned int>(image_f.size().height), static_cast<unsigned int>(image_f.size().width) }, ie::Layout::NHWC);
    ireq_pd.SetBlob(input_name_pd, ie::make_shared_blob<float>(tDesc, (float*)(image_f.data)));

    ireq_pd.Infer();                                                            // Pose detection
    float* scores = ireq_pd.GetBlob(oname_pd[0])->buffer();
    float* bboxes = ireq_pd.GetBlob(oname_pd[1])->buffer();

    std::list<detect_region_t> region_list, region_nms_list;
    std::vector<detect_region_t> detect_results;
    decode_bounds(region_list, 0.5f, idims_pd[_W], idims_pd[_H], scores, bboxes, anchors);
    non_max_suppression(region_list, region_nms_list, 0.7f);
    pack_detect_result(detect_results, region_nms_list, idims_pd);


    // landmark detection
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

        img_affine.convertTo(image_f, CV_32F, (1 / 127.5f), -1.f);
        tDesc = ie::TensorDesc(ie::Precision::FP32, { 1, (unsigned int)(image_f.channels()), (unsigned int)(image_f.size().height), (unsigned int)(image_f.size().width) }, ie::Layout::NHWC);
        ireq_lm.SetBlob(input_name_lm, ie::make_shared_blob<float>(tDesc, (float*)(image_f.data)));

        ireq_lm.Infer();                                                // Landmark detection

        float* output_lm1 = ireq_lm.GetBlob(oname_lm[0])->buffer();     // ??
        float* lm_score   = ireq_lm.GetBlob(oname_lm[1])->buffer();     // Landmark score (flag)
        float* lm         = ireq_lm.GetBlob(oname_lm[2])->buffer();     // Landmarks

        landmarks[pose_id].score = *lm_score;
        for (size_t i = 0; i < POSE_JOINT_NUM; i++) {
            landmarks[pose_id].joint[i].x = lm[4 * i + 0] / (float)idims_lm[_W];
            landmarks[pose_id].joint[i].y = lm[4 * i + 1] / (float)idims_lm[_H];
            landmarks[pose_id].joint[i].z = lm[4 * i + 2];
        }
        renderROI(image_ocv, pose.roi_coord);
        renderPose(image_ocv, detect_results, landmarks);
    }
    cv::imwrite("output.jpg", image_ocv);
    cv::imshow("output", image_ocv);
    cv::waitKey(0);

    return 0;
}

#if 0    // cheat memo
id :0
Score : 0.80571, topleft : (0.599791, 0.322176) btmright(0.803113, 0.525553)
keys : (0.592301, 0.60807) (0.765157, 0.279203) (0.694426, 0.464116) (0.794771, 0.27079)
rotation(0.617028, roi_center(0.694426, 0.464116), roi_size(0.65345, 0.65345)
roi_coord[0] = (0.616996, 0.00859112) roi_coord[1] = (1.14995, 0.386686) roi_coord[2] = (0.771856, 0.919642) roi_coord[3] = (0.238901, 0.541547)


typedef struct _detect_region_t
{
    float score;
    fvec2 topleft;
    fvec2 btmright;
    fvec2 keys[kPoseDetectKeyNum];

    float  rotation;
    fvec2  roi_center;
    fvec2  roi_size;
    fvec2  roi_coord[4];
} detect_region_t;

struct _pose_detect_result_t
{
    int num;
    detect_region_t poses[MAX_POSE_NUM];
}
id:0
Score : 0.80571, topleft : (0.599791, 0.322176) btmright(0.803113, 0.525553)
keys : (0.592301, 0.60807) (0.765157, 0.279203) (0.694426, 0.464116) (0.794771, 0.27079)
rotation(0.617028, roi_center(0.694426, 0.464116), roi_size(0.65345, 0.65345)
    roi_coord[0] = (0.616996, 0.00859112) roi_coord[1] = (1.14995, 0.386686) roi_coord[2] = (0.771856, 0.919642) roi_coord[3] = (0.238901, 0.541547)
    103.5, 99.6409, -0.118641
    106.968, 95.2421, -0.599489
    107.992, 95.073, -0.879786
    109.055, 94.9266, 0.589533
    106.99, 95.3374, -0.63118
    108.001, 95.3395, -0.185324
    108.963, 95.3678, 1.58586
    116.709, 96.6343, 1.33967
    116.567, 96.885, 1.53691
    107.784, 104.388, -0.353246
    107.213, 104.764, 0.94916
    123.701, 123.332, -67.1285
    122.666, 128.015, 64.7882
    80.9464, 139.919, -90.088
    83.8278, 144.983, 53.8685
    41.4329, 147.431, 96.8092
    49.1288, 150.342, -43.2139
    32.8085, 145.617, 0.113534
    43.4589, 146.909, -0.993368
    33.1729, 142.45, -1.34155
    42.0035, 144.318, 0.271668
    35.7591, 145.685, 0.440072
    43.3881, 146.493, 0.722159
    126.391, 183.047, -5.03068
    121.725, 179.694, 124.67
#endif  
