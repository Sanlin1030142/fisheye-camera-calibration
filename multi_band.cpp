
#include <cv_bridge/cv_bridge.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <vector>

#define partial 4

using namespace std;
using namespace cv;

std::vector<cv::Mat> frames(6);

void frameCallback(const sensor_msgs::ImageConstPtr& msg, int index) {
    // cout << index << endl;
    frames[index] = cv_bridge::toCvCopy(msg, "bgr8")->image;
}

vector<Mat_<float>> gaussianPyramid(const Mat& img, int levels) {
    vector<Mat_<float>> _gaussianPyramid;
    _gaussianPyramid.push_back(img);
    Mat currentImage = img;
    // cout << currentImage.size() << endl;
    for (int i = 1; i < levels; i++) {
        Mat downsampleImage;
        pyrDown(currentImage, downsampleImage);  // Blurs an image and downsamples it.
        _gaussianPyramid.push_back(downsampleImage);
        currentImage = downsampleImage;
    }
    return _gaussianPyramid;
}

vector<Mat_<float>> laplacianPyramid(const vector<Mat_<float>>& gaussPyrImg) {
    int levels = gaussPyrImg.size();
    vector<Mat_<float>> _laplacianPyramid;
    _laplacianPyramid.push_back(gaussPyrImg[levels - 1]);  // order reverse !!
    // cout << gaussPyrImg[levels - 1].size() << endl;
    for (int i = levels - 2; i >= 0; i--) {
        Mat upsampleImage;
        pyrUp(gaussPyrImg[i + 1], upsampleImage, gaussPyrImg[i].size());
        Mat currentImage = gaussPyrImg[i] - upsampleImage;
        _laplacianPyramid.push_back(currentImage);
    }
    return _laplacianPyramid;
}

vector<Mat_<float>> blendPyramid(const vector<Mat_<float>>& pyrA, const vector<Mat_<float>>& pyrB, const vector<Mat_<float>>& pyrMask) {
    int levels = pyrA.size();
    vector<Mat_<float>> blendedPyramid;
    for (int i = 0; i < levels; i++) {
        Mat blendedImage = pyrA[i].mul(1.0 - pyrMask[levels - 1 - i]) + pyrB[i].mul(pyrMask[levels - 1 - i]);
        blendedPyramid.push_back(blendedImage);
    }
    return blendedPyramid;
}

Mat collapsePyramid(const vector<Mat_<float>>& blendedPyramid) {
    int levels = blendedPyramid.size();
    Mat currentImage = blendedPyramid[0];
    for (int i = 1; i < levels; i++) {
        pyrUp(currentImage, currentImage, blendedPyramid[i].size());
        currentImage += blendedPyramid[i];
    }
    Mat blendedImage;
    convertScaleAbs(currentImage, blendedImage, 255.0);
    return blendedImage;
}

Mat merge_two_frame(Mat& A, Mat& B) {
    if (A.size() != B.size()) {
        A = Mat(A, Range::all(), Range(0, A.cols));
        B = Mat(B, Range(0, A.rows), Range(0, A.cols));
    }

    int height = A.rows;
    int width = A.cols;

    // Convert images to float
    Mat imgA, imgB;
    A.convertTo(imgA, CV_32F, 1.0 / 255.0);
    B.convertTo(imgB, CV_32F, 1.0 / 255.0);

    // Create mask
    Mat_<float> mask(height, width, 0.0);
    mask(Range::all(), Range(mask.cols / 2 + 2, mask.cols)) = 1.0;

    // Create gaussian pyramids for the mask
    int levels = floor(log2(min(width, height)));
    vector<Mat_<float>> gaussPyrMask = move(gaussianPyramid(mask, levels));

    // Create and blend pyramids for each channel
    vector<Mat> channelsA, channelsB, blendedChannels;

    split(imgA, channelsA);
    split(imgB, channelsB);

    for (int c = 0; c < 3; c++) {
        vector<Mat_<float>> gaussPyrA = move(gaussianPyramid(channelsA[c], levels));
        vector<Mat_<float>> gaussPyrB = move(gaussianPyramid(channelsB[c], levels));
        vector<Mat_<float>> laplacePyrA = move(laplacianPyramid(gaussPyrA));
        vector<Mat_<float>> laplacePyrB = move(laplacianPyramid(gaussPyrB));
        vector<Mat_<float>> blendedPyr = move(blendPyramid(laplacePyrA, laplacePyrB, gaussPyrMask));
        Mat blendedChannel = move(collapsePyramid(blendedPyr));
        blendedChannel.convertTo(blendedChannel, CV_32F, 1.0 / 255.0);  // Ensure the range is [0, 1]
        blendedChannels.push_back(blendedChannel);
    }

    Mat blendedImg;
    merge(blendedChannels, blendedImg);
    blendedImg.convertTo(blendedImg, CV_8UC3, 255.0);
    return blendedImg;
}

Mat merge_overlapping(Mat& A, Mat& B) {
    int height = A.rows;
    int width = A.cols;
    int overlap_width = width / partial;
    Mat overlap_A = A(cv::Rect(width - overlap_width, 0, overlap_width, height));
    Mat overlap_B = B(cv::Rect(0, 0, overlap_width, height));
    return merge_two_frame(overlap_A, overlap_B);
}

int main(int argc, char** argv) {
    // Read the images, assuming images to be of same size

    ros::init(argc, argv, "stitched_frame");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<sensor_msgs::Image>("stitched_frame", 1);

    ros::Subscriber sub0 = nh.subscribe<sensor_msgs::Image>("/trace/camera_0", 1, boost::bind(frameCallback, _1, 0));
    ros::Subscriber sub1 = nh.subscribe<sensor_msgs::Image>("/trace/camera_1", 1, boost::bind(frameCallback, _1, 1));
    ros::Subscriber sub2 = nh.subscribe<sensor_msgs::Image>("/trace/camera_2", 1, boost::bind(frameCallback, _1, 2));
    ros::Subscriber sub3 = nh.subscribe<sensor_msgs::Image>("/trace/camera_3", 1, boost::bind(frameCallback, _1, 3));
    ros::Subscriber sub4 = nh.subscribe<sensor_msgs::Image>("/trace/camera_4", 1, boost::bind(frameCallback, _1, 4));
    ros::Subscriber sub5 = nh.subscribe<sensor_msgs::Image>("/trace/camera_5", 1, boost::bind(frameCallback, _1, 5));

    ros::Rate loop_rate(40);

    while (ros::ok()) {
        ros::spinOnce();

        if (frames[0].empty() || frames[1].empty()|| frames[2].empty()|| frames[3].empty()|| frames[4].empty()|| frames[5].empty()) {
            continue;
        }

        // Single frame size
        int width = frames[0].cols;
        int height = frames[0].rows;
        int overlap_width = width / partial;
        Mat panorama(288, 2112, frames[0].type(), cv::Scalar::all(0));

        panorama = frames[0](cv::Rect(overlap_width, 0, width - (2 * overlap_width) , height));

        for ( int i = 1 ; i < 6 ; i++ ) {
            Mat overlapping = move(merge_overlapping(frames[i-1], frames[i]));
            cv::hconcat(panorama, overlapping, panorama);
            cv::hconcat(panorama, frames[i](cv::Rect(overlap_width, 0, width - ( 2 * overlap_width ), height)), panorama);
        } 

        Mat overlapping = move(merge_overlapping(frames[5], frames[0]));
        cv::hconcat(panorama, overlapping, panorama);

        
        

        if (!panorama.empty()) {
            // Publish merged frame
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", panorama).toImageMsg();
            pub.publish(msg);
        }

        loop_rate.sleep();
    }

    return 0;
}