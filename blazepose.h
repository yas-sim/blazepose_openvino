#pragma once
/* ------------------------------------------------ *
 * The MIT License (MIT)
 * Copyright (c) 2020 terryky1220@gmail.com
 * ------------------------------------------------ */
#ifndef TFLITE_BLAZEPOSE_H_
#define TFLITE_BLAZEPOSE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_POSE_NUM  100

#ifndef FULL_POSE
#define POSE_JOINT_NUM 25
#else
#define POSE_JOINT_NUM 33
#endif

#ifndef FULL_POSE
    enum pose_detect_key_id {
        kMidHipCenter = 0,      //  0
        kFullBodySizeRot,       //  1
        kMidShoulderCenter,     //  2
        kUpperBodySizeRot,      //  3

        kPoseDetectKeyNum
    };
#else
    enum pose_detect_key_id {
        kMidHipCenter = 0,      //  0
        kFullBodySizeRot,       //  1

        kPoseDetectKeyNum
    };
#endif

    typedef struct _fvec2
    {
        float x, y;
    } fvec2;

    typedef struct _fvec3
    {
        float x, y, z;
    } fvec3;

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

    typedef struct _pose_detect_result_t
    {
        int num;
        detect_region_t poses[MAX_POSE_NUM];
    } pose_detect_result_t;

    typedef struct _pose_landmark_result_t
    {
        float score;
        fvec3 joint[POSE_JOINT_NUM];
    } pose_landmark_result_t;


    typedef struct _blazepose_config_t
    {
        float score_thresh;
        float iou_thresh;
    } blazepose_config_t;

    int init_tflite_blazepose(int use_quantized_tflite, blazepose_config_t* config);

    void* get_pose_detect_input_buf(int* w, int* h);
    int  invoke_pose_detect(pose_detect_result_t* detect_result, blazepose_config_t* config);

    void* get_pose_landmark_input_buf(int* w, int* h);
    int  invoke_pose_landmark(pose_landmark_result_t* pose_landmark_result);

#ifdef __cplusplus
}

typedef struct Anchor
{
    float x_center, y_center, w, h;
};

typedef struct SsdAnchorsCalculatorOptions
{
    // Size of input images.
    int input_size_width;                   // [required]
    int input_size_height;                  // [required]

    // Min and max scales for generating anchor boxes on feature maps.
    float min_scale;                        // [required]
    float max_scale;                        // [required]

    // The offset for the center of anchors. The value is in the scale of stride.
    // E.g. 0.5 meaning 0.5 * |current_stride| in pixels.
    float anchor_offset_x;                  // default = 0.5
    float anchor_offset_y;                  // default = 0.5

    // Number of output feature maps to generate the anchors on.
    int num_layers;                         // [required]

    // Sizes of output feature maps to create anchors. Either feature_map size or
    // stride should be provided.
    std::vector<int> feature_map_width;
    std::vector<int> feature_map_height;

    // Strides of each output feature maps.
    std::vector<int>   strides;

    // List of different aspect ratio to generate anchors.
    std::vector<float> aspect_ratios;

    // A boolean to indicate whether the fixed 3 boxes per location is used in the
    // lowest layer.
    bool reduce_boxes_in_lowest_layer;      // default = false

    // An additional anchor is added with this aspect ratio and a scale
    // interpolated between the scale for a layer and the scale for the next layer
    // (1.0 for the last layer). This anchor is not included if this value is 0.
    float interpolated_scale_aspect_ratio;  // default = 1.0

    // Whether use fixed width and height (e.g. both 1.0f) for each anchor.
    // This option can be used when the predicted anchor width and height are in
    // pixels.
    bool fixed_anchor_size;                 // default = false
} SsdAnchorsCalculatorOptions_t;

#endif

#endif /* TFLITE_BLAZEPOSE_H_ */
