#pragma once

# include<opencv2/highgui/highgui.hpp>
# include<opencv2/imgproc/imgproc.hpp>
# include<opencv2/opencv.hpp>
# include<iostream>

using namespace std;
using namespace cv;

class UOvec
{
public:
	UOvec();
	~UOvec();

	vector<Rect> objects;

	void add(vector<Rect> newObjects);
};

