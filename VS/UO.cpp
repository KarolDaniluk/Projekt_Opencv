#include "UO.h"

#include "utility.h"

UO::UO()
{
}


UO::~UO()
{
}

void UO::init(Rect rectangle, Mat frame)
{
	rect = rectangle;
	time_init = time(0);
	frame.copyTo(frame_init);
	//pos_init = center(rect);
	initialized = true;
	if (print) cout << "INIT" << endl;
}

void UO::activate(Mat frame)
{
	time_active = time(0);
	frame.copyTo(frame_active);
	//pos_active = center(rect);
	getHOGdetector();
	activated = true;
	if (print) cout << "ACTIVATE" << endl;
}

void UO::alarm(Mat frame)
{
	time_alarm = time(0);
	frame.copyTo(frame_alarm);
	//pos_alarm = center(rect);
	alarmed = true;
	if (print) cout << "ALARM" << endl;
	imshow("ALERT - init image", frame_init);
	imshow("ALERT - activ image", frame_active);
	imshow("ALERT - alarm image", frame_alarm);
}

void UO::checkState(Rect rectangle, Mat frame)
{
	if (!alarmed && activeTime() > alarm_time)
		alarm(frame);
	else if (!activated && initTime() > active_time)
		activate(frame);
	else if (!initialized)
		init(rectangle, frame);
	//else if (activated)
		//detectHumans(frame, 1);
}


double UO::humanDist()
{
	return 12.0;
}

void UO::update(vector<Rect> rectangles, Mat frame, bool releaseIfEmpty)
{
	Rect rectangle = overlaping(rectangles);
	if (rectangle.empty() && initialized && releaseIfEmpty)
		release();
	else if (!rectangle.empty())
	{
		frame.copyTo(frame_last);
		checkState(rectangle, frame);
		distTracked += calcDist(rectangle);
		rect = rectangle;
	}
}

Rect UO::overlaping(vector<Rect> rectangles)
{
	if (rectangles.size() == 0)
		return Rect();
	else if (!initialized)
		return rectangles[0];
	Rect tmp;
	for (int i = 0; i < rectangles.size(); i++)
	{
		if ((rect & rectangles[i]).area() > 0)
			tmp = rectangles[i];
	}
	return tmp;
}

bool UO::detectHumans(Mat frame, int expandScale)
{
	cuda::GpuMat frameG;
	frameG.upload(frame);
	cuda::cvtColor(frameG, frameG, COLOR_BGR2GRAY);
	Rect roi = scaleRect(rect, expandScale);
	Point offset = Point(roi.x, roi.y);
	vector<Rect> tmp;
	detector->detectMultiScale(frameG(roi), tmp);
	for (int i = 0; i < tmp.size(); i++)
		tmp[i] += offset;
	if (tmp.size() > 0)
	{
		humans = tmp;
		return true;
	}
	else
		return false;
}

void UO::getHOGdetector()
{
	detector = cuda::HOG::create();
	Mat hog_pedestrians = detector->getDefaultPeopleDetector();	// ³adowanie wbudowanego modelu
	detector->setSVMDetector(hog_pedestrians);					// przypisanie modelu do klasyfikatora
	hog_pedestrians.release();
}


int UO::get_state()
{
	if (alarmed)
		return 3;
	else if (activated)
		return 2;
	else if (initialized)
		return 1;
	else return 0;
}

void UO::release()
{
	initialized = false;
	activated = false;
	alarmed = false;
	rect = Rect();
	time_init = 0;
	time_active = 0;
	time_alarm = 0;
	frame_last.release();
	frame_init.release();
	frame_active.release();
	frame_alarm.release();
	detector.release();
	//pos_init = Point();
	//pos_active = Point();
	//pos_alarm = Point();
	distTracked = 0;
	if (print) cout << "RELEASE" << endl;	///
}


double UO::calcDist(Rect rectangle)
{
	Point A = center();
	Point B = center(rectangle);
	int dx = abs(A.x - B.x);
	int dy = abs(A.y - B.y);
	double dist = sqrt((double)(pow(dx, 2) + pow(dy, 2)));
	return dist;
}

double UO::calcDist(Rect rectangle1, Rect rectangle2)
{
	Point A = center(rectangle1);
	Point B = center(rectangle2);
	int dx = abs(A.x - B.x);
	int dy = abs(A.y - B.y);
	double dist = sqrt((double)(pow(dx, 2) + pow(dy, 2)));
	return dist;
}

int UO::initTime()
{
	if (initialized)
		return (time(0) - time_init);
	else
		return 0;
}

int UO::activeTime()
{
	if (activated)
		return (time(0) - time_active);
	else
		return 0;
}

int UO::alarmTime()
{
	if (alarmed)
		return (time(0) - time_alarm);
	else
		return 0;
}

Point UO::center()
{
	return Point(rect.x + rect.width / 2, rect.y + rect.height / 2);
}

Point UO::center(Rect rectangle)
{
	return Point(rectangle.x + rectangle.width / 2, rectangle.y + rectangle.height / 2);
}

Point UO::position()
{
	return Point(rect.x, rect.y);
}

Scalar UO::color()
{
	if (alarmed)
		return Scalar(0, 0, 255);
	else if (activated)
		return Scalar(0, 255, 0);
	else
		return Scalar(255, 255, 255);
}

int UO::thickness()
{
	if (alarmed)
		return 2;
	else
		return 1;
}

//Rect UO::overlaping(vector<Rect> rectangles)
//{
//	if (rectangles.size() == 0)
//		return Rect();
//	else if (!initialized)
//		return rectangles[0];
//	int counter = 0;
//	Rect tmp;
//	for (int i = 0; i < rectangles.size(); i++)
//	{
//		if ((rect & rectangles[i]).area() > 0)
//		{
//			tmp = rectangles[i];
//			counter++;
//		}
//	}
//	if (counter > 1)
//		int stop = 0;
//	return tmp;
//}