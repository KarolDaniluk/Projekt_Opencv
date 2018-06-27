# pragma once

# include<opencv2/highgui/highgui.hpp>
# include<opencv2/imgproc/imgproc.hpp>
# include<opencv2/opencv.hpp>
# include<iostream>

using namespace std;
using namespace cv;

class Tracked
{
public:
	// Creators
	Tracked();							// Default empty creator
	Tracked(vector<Point> contour);		// Standard creator
	Tracked(vector<Point> contour, double smoothLevel);		// Creator with instant shape smoothing

	// Destroyer
	~Tracked();

	// Getters
	vector<Point> shape();	// Shape of the object
	double area();				// Area of the object
	Point centre();				// Centre of mass		
	double x();					// X of mass centre
	double y();					// Y of mass centre
	vector<Point> smoothed(double smoothLevel);	// Return smoothed shape (no update)
	Rect bounding();
	vector<Point> convex();

	// Setters
	void shape(vector<Point> contour);	// Set shape
	void smooth(double smoothLevel);	// Smooth shape of current object (update)

private:

	// Variables
	vector<Point> contour;	// Shape of the object
	bool isMoment = false;	// True if moment is up to date
	Moments moment;			// Moments of the shape

	// Methods
	void updateMoment();	// Upate moments of the shape
};
