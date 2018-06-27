# include "Tracked.h"

// DESTROYER
Tracked::~Tracked() {}

// CREATORS
// Default empty creator
Tracked::Tracked() {}

// Standard creator
Tracked::Tracked(vector<Point> contour)
{
	this->shape(contour);
}

// Creator with instant shape smoothing
Tracked::Tracked(vector<Point> contour, double smoothLevel)
{
	this->shape(contour);
	this->smooth(smoothLevel);
}


// GETTERs
// Shape of the object
vector<Point> Tracked::shape()
{
	return contour;
}

// Area of the object
double Tracked::area()
{
	return contourArea(contour);
}

// Centre of mass
Point Tracked::centre()
{
	this->updateMoment();
	double x = moment.m10 / moment.m00;
	double y = moment.m01 / moment.m00;
	return(Point(x, y));
}

// X of mass centre
double Tracked::x()
{
	this->updateMoment();
	return(moment.m10 / moment.m00);
}

// Y of mass centre
double Tracked::y()
{
	this->updateMoment();
	return(moment.m01 / moment.m00);
}

// Return smoothed shape
vector<Point> Tracked::smoothed(double smoothLevel)
{
	vector<Point> tmp = this->contour;
	double eps = smoothLevel * arcLength(tmp, true);
	approxPolyDP(tmp, tmp, eps, true);
	return(tmp);
}

Rect Tracked::bounding()
{
	return(boundingRect(contour));
}

vector<Point> Tracked::convex()
{
	Mat hull;
	convexHull(contour, hull);
	return(hull);
}

// SETTERs
// Set shape
void Tracked::shape(vector<Point> contour)
{
	this->contour = contour;
	isMoment = false;
}

// Smooth shape of current object
void Tracked::smooth(double smoothLevel)
{
	this->shape(this->smoothed(smoothLevel));
}

// OTHER
// Upate moments of the shape
void Tracked::updateMoment()
{
	if (!isMoment)
	{
		this->moment = moments(this->contour);
		this->isMoment = true;
	}
}