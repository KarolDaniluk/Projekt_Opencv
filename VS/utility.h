#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <conio.h>
//#include "HSVfilter.h"
#include "Tracked.h"

using namespace std;
using namespace cv;

double round2dec(double value, int decimals = 2);

void blendImages(Mat &output, Mat img1, Mat img2, Mat mask);

void getRecorder(VideoCapture &cap, double &delay, string source, bool &isVideo);
void getRecorder(VideoCapture &cap, double &delay, string source);

void restrictValue(int &value, int min, int max, int step);
void restrictValue(double &value, double min, double max, double step);
void restrictValue(int &value, vector<int> numbers);

bool sortContours(vector<vector<Point>> contoursToSort, vector<vector<Point>> &contoursSorted, double minArea = 0);
bool sortContours(vector<vector<Point>> contoursToSort, vector<vector<Point>> &contoursSorted, vector<double> &areas, double minArea = 0);

void cuda_accumulateWeighted(cv::cuda::GpuMat src, cv::cuda::GpuMat &dst, double alpha, cv::cuda::Stream stream = cv::cuda::Stream::Null());

void putTextBetter(Mat &input, string text, string position = "TopLeft", double size = 1, Scalar color = Scalar(255, 255, 255), int thickness = 1, int font = 0, int type = 8);
void putTextBetter(Mat &input, string text, string position, Point offset, double size = 1, Scalar textColor = Scalar(255, 255, 255), int thickness = 1, int font = 0, int type = 8);
void putTextBetter(Mat &input, string text, string position, double size, Scalar textColor, Scalar backgroundColor, int thickness = 1, int font = 0, int type = 8);
void putTextBetter(Mat &input, string text, string position, Point offset, double size, Scalar textColor, Scalar backgroundColor, int thickness = 1, int font = 0, int type = 8);

string frame2time(int frameNum, double fps);

bool overlapingContours(Mat img1, Mat img2, vector<Rect> &outputRectangles, double minArea = 0);

Rect scaleRect(Rect input, double scale);
Rect scaleRect(Rect input, double xScale, double yScale);
Rect scaleRect(Rect input, Size inputImgSize, Size outputImgSize);
