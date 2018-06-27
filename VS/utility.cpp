#include "utility.h"

// round value to decimals places
double round2dec(double value, int decimals)
{
	double divider = pow(10, decimals);
	return(round(value * divider) / divider);
}

// restrict value to the closest one between min and max range and given step
void restrictValue(double &value, double min, double max, double step)
{
	if (value < min)
		value = min;
	else if (value > max)
		value = max;
	else if (fmod(value, step) != min)
	{
		int prev = min + step * (floor(value / step));
		int next = min + step * (floor(value / step) + 1);
		double avg = ((prev + next) / 2);
		if (value >= avg)
			value = next;
		else
			value = prev;
	}
}

// restrict value to the closest one between min and max range and given step (int overload)
void restrictValue(int &value, int min, int max, int step)
{
	if (value < min)
		value = min;
	else if (value > max)
		value = max;
	else if (value % step != min)
	{
		int prev = min + step * (floor(value / step));
		int next = min + step * (floor(value / step) + 1);
		double avg = ((prev + next) / 2);
		if (value >= avg)
			value = next;
		else
			value = prev;
	}
}

// restrict value to the closest one in vector
void restrictValue(int &value, vector<int> numbers)
{
	int diff = abs(numbers[0] - value);
	int id = 0;
	for (int i = 0; i < numbers.size(); i++)
	{
		if (abs(numbers[i] - value) < diff)
		{
			id = i;
			diff = abs(numbers[i] - value);
		}
	}
	value = numbers[id];
}


// blend two equally sized images using transparency mask
void blendImages(Mat &output, Mat img1, Mat img2, Mat mask)
{
	// check if images sizes match
	if (img1.size() != img2.size())
		cout << "Error. Images size does not match" << endl;

	// check if number of channels match
	if (img1.channels() != img2.channels())
		cout << "Error. Number of channels in images does not match" << endl;
	
	// mask normalization
	double maxi;
	minMaxLoc(mask, NULL, &maxi);
	if (maxi > 1)
		mask = mask / maxi;

	// check image and mask channels
	int C = img1.channels();
	int M = mask.channels();

	// M x 1 Scalar
	Scalar Ones;

	// C channels mask
	if (M != C || M == 1)
	{
		Mat *dummy = new Mat[C];

		for (int i = 0; i < C; i++)
		{
			dummy[i] = mask;
			Ones[i] = 1;
		}
		merge(dummy, C, mask);
	}
	else
		cout << "Error. Number of mask channels is different than 1 or images channels" << endl;

	// blending images
	output = img1.mul(Ones - mask) + img2.mul(mask);
}


// obtain video capure object class and proper delay (waitKey) from source as device (eg. string "0") or path to the video file
void getRecorder(VideoCapture &cap, double &delay, string source, bool &isVideo)
{
	delay = 40;
	// spróbuj odczytaæ Ÿród³o jako kamerê
	try
	{
		int device = stoi(source);
		if (device >= 0)
		{
			cap.open(device);
			cout << "Video capture initialization with device = " << device << " ..." << endl;
		}
		else
			cout << "Video capture device could not be read as negative value." << endl;
	}
	// odczytaj Ÿród³o jako œcie¿kê do pliku
	catch (...)
	{
		cap.open(source);
		delay = 1000 / cap.get(CV_CAP_PROP_FPS);
		cout << "Video capture initialization with path = " << source << "..." << endl;
	}

	// sprawdŸ czy uda³o siê odczytaæ plik
	if (cap.isOpened())
	{
		isVideo = true;
		cout << "Video capture initialized successfully." << endl;
	}
	else
	{
		isVideo = false;
		cout << "Video capture cannot be read as Camera or video file." << endl;
	}
}


void getRecorder(VideoCapture &cap, double &delay, string source)
{
	delay = 40;
	// spróbuj odczytaæ Ÿród³o jako kamerê
	try
	{
		int device = stoi(source);
		if (device >= 0)
		{
			cap.open(device);
			cout << "Video capture initialization with device = " << device << " ..." << endl;
		}
		else
			cout << "Video capture device could not be read as negative value." << endl;
	}
	// odczytaj Ÿród³o jako œcie¿kê do pliku
	catch (...)
	{
		cap.open(source);
		delay = 1000 / cap.get(CV_CAP_PROP_FPS);
		cout << "Video capture initialization with path = " << source << "..." << endl;
	}

	// sprawdŸ czy uda³o siê odczytaæ plik
	if (cap.isOpened())
		cout << "Video capture initialized successfully." << endl;
	else
		cout << "Video capture cannot be read as Camera or video file." << endl;
}


// compare first values of two pair object
bool compare1(const pair<int, int>&i, const pair<int, int>&j)
{
	return i.first < j.first;
}

// compare second values of two pair object 
bool compare2(const pair<int, int>&i, const pair<int, int>&j)
{
	return i.second < j.second;
}

// sort descending contours by area, above given threshold (minArea)
void sortContoursX(vector<vector<Point>> contoursToSort, vector<vector<Point>> &contoursSorted, double minArea)
{
	int m = contoursToSort.size();	// liczba konturów

	vector<pair<double, int>> tab;	// tab = [area, index]

	// obliczanie powierzchni ka¿dego konturu
	for (int i = 0; i < m; i++)
	{
		double a = contourArea(contoursToSort[i]);	// pole powierzchni i-tego konturu
		if (a >= minArea)							// je¿eli powierzchnia i-tego konturu jest wiêksza ni¿ minimalne pole powierzchni (minArea)
			tab.push_back(make_pair(a, i));			// dodaj go do tablicy tab
	}

	sort(tab.rbegin(), tab.rend(), compare1);		// posortuj po polu powierzchni od najwiekszego do najmniejszego

	int n = tab.size();			// zakfalifikowana liczba konturów
	contoursSorted.clear();		// wyczyœæ zmienn¹ konturów
	for (int i = 0; i < n; i++)
	{
		int idx = tab[i].second;	// indeks i-tego najwiêkszego konturu w zmiennej contoursToSort
		contoursSorted.push_back(contoursToSort[idx]);	// przypis ten kontur do zmiennej contoursSorted
	}
}

// sort descending contours by area, above given threshold (minArea)
bool sortContours(vector<vector<Point>> contoursToSort, vector<vector<Point>> &contoursSorted, double minArea)
{
	vector<double> dummy;
	bool ans = sortContours(contoursToSort, contoursSorted, dummy, minArea);
	return ans;
}

// sort descending contours by area, above given threshold (minArea) + return vector of their areas
bool sortContours(vector<vector<Point>> contoursToSort, vector<vector<Point>> &contoursSorted, vector<double> &areas, double minArea)
{
	if (contoursToSort.size() == 0)
		return false;
	int m = contoursToSort.size();	// liczba konturów
	vector<pair<double, int>> tab;	// tab = [area, index]
	for (int i = 0; i < m; i++)		// oblicz pole powierzchni ka¿dego konturu
	{
		double a = contourArea(contoursToSort[i]);	// pole powierzchni i-tego konturu
		if (a >= minArea)							// je¿eli powierzchnia i-tego konturu jest wiêksza ni¿ minimalne pole powierzchni (minArea)
			tab.push_back(make_pair(a, i));			// dodaj go do tablicy tab
	}
	sort(tab.rbegin(), tab.rend(), compare1);	// posortuj po polu powierzchni od najwiekszego do najmniejszegos
	int n = tab.size();			// zakfalifikowana liczba konturów
	contoursSorted.clear();		// wyczyœæ zmienn¹ konturów
	areas.clear();				// wyczyœæ zmienn¹ pól powierzchni
	for (int i = 0; i < n; i++)
	{
		int idx = tab[i].second;	// indeks i-tego najwiêkszego konturu w zmiennej contoursToSort
		contoursSorted.push_back(contoursToSort[idx]);	// dopisz ten kontur do zmiennej contoursSorted
		areas.push_back(tab[i].first);	// dopisz pole powierzchni
	}
	if (contoursSorted.size() > 0)
		return true;
	else
		return false;
}

// Cuda version of accumulateWeighted function
void cuda_accumulateWeighted(cv::cuda::GpuMat src, cv::cuda::GpuMat &dst, double alpha, cv::cuda::Stream stream)
{
	cuda::GpuMat result;
	double beta = 1.0 - alpha;
	cuda::addWeighted(dst, beta, src, alpha, 0, result, -1, stream);
	dst = result;
}

// Put text on image with position given by string and default color and font
void putTextBetter(Mat &input, string text, string position, double size, Scalar color, int thickness, int font, int type)
{
	transform(position.begin(), position.end(), position.begin(), ::toupper);	// zamiana na wielkie litery
	Size frameSize = input.size();
	int W = frameSize.width;
	int H = frameSize.height;
	Size textSize = getTextSize(text, font, size, thickness, 0);
	int h = textSize.height;
	int w = textSize.width;
	Point pos;
	if (position == "TL" || position == "TOPLEFT")
		pos = Point(0, h);
	else if (position == "TR" || position == "TOPRIGHT")
		pos = Point(W - w, h);
	else if (position == "BL" || position == "BOTTOMLEFT")
		pos = Point(0, H);
	else if (position == "BR" || position == "BOTTOMRIGHT")
		pos = Point(W - w, H);
	else if (position == "T" || position == "TOP")
		pos = Point((W - w) / 2, h);
	else if (position == "L" || position == "LEFT")
		pos = Point(0, (H + h) / 2);
	else if (position == "R" || position == "RIGHT")
		pos = Point(W - w, (H + h) / 2);
	else if (position == "B" || position == "BOTTOM")
		pos = Point((W - w) / 2, H);
	else if (position == "C" || position == "CENTER")
		pos = Point((W - w) / 2, (H + h) / 2);
	else
		pos = Point(0, h);
	putText(input, text, pos, font, size, color, thickness, type, false);
}

void putTextBetter(Mat &input, string text, string position, Point offset, double size, Scalar textColor, int thickness, int font, int type)
{
	transform(position.begin(), position.end(), position.begin(), ::toupper);	// zamiana na wielkie litery
	Size frameSize = input.size();
	int W = frameSize.width;
	int H = frameSize.height;
	Size textSize = getTextSize(text, font, size, thickness, 0);
	int h = textSize.height;
	int w = textSize.width;
	Point pos;
	if (position == "TL" || position == "TOPLEFT")
		pos = Point(0, h) + offset;
	else if (position == "TR" || position == "TOPRIGHT")
		pos = Point(W - w, h) + offset;
	else if (position == "BL" || position == "BOTTOMLEFT")
		pos = Point(0, H) + offset;
	else if (position == "BR" || position == "BOTTOMRIGHT")
		pos = Point(W - w, H) + offset;
	else if (position == "T" || position == "TOP")
		pos = Point((W - w) / 2, h) + offset;
	else if (position == "L" || position == "LEFT")
		pos = Point(0, (H + h) / 2) + offset;
	else if (position == "R" || position == "RIGHT")
		pos = Point(W - w, (H + h) / 2) + offset;
	else if (position == "B" || position == "BOTTOM")
		pos = Point((W - w) / 2, H) + offset;
	else if (position == "C" || position == "CENTER")
		pos = Point((W - w) / 2, (H + h) / 2) + offset;
	else
		pos = Point(0, h) + offset;
	putText(input, text, pos, font, size, textColor, thickness, type, false);
}

void putTextBetter(Mat &input, string text, string position, double size, Scalar textColor, Scalar backgroundColor, int thickness, int font, int type)
{
	transform(position.begin(), position.end(), position.begin(), ::toupper);	// zamiana na wielkie litery
	Size frameSize = input.size();
	int W = frameSize.width;
	int H = frameSize.height;
	Size textSize = getTextSize(text, font, size, thickness, 0);
	int h = textSize.height;
	int w = textSize.width;
	Point pos;
	if (position == "TL" || position == "TOPLEFT")
		pos = Point(0, h);
	else if (position == "TR" || position == "TOPRIGHT")
		pos = Point(W - w, h);
	else if (position == "BL" || position == "BOTTOMLEFT")
		pos = Point(0, H);
	else if (position == "BR" || position == "BOTTOMRIGHT")
		pos = Point(W - w, H);
	else if (position == "T" || position == "TOP")
		pos = Point((W - w) / 2, h);
	else if (position == "L" || position == "LEFT")
		pos = Point(0, (H + h) / 2);
	else if (position == "R" || position == "RIGHT")
		pos = Point(W - w, (H + h) / 2);
	else if (position == "B" || position == "BOTTOM")
		pos = Point((W - w) / 2, H);
	else if (position == "C" || position == "CENTER")
		pos = Point((W - w) / 2, (H + h) / 2);
	else
		pos = Point(0, h);
	Rect RoI = Rect(pos, pos + Point(w, -h));
	Mat background = Mat(RoI.height, RoI.width, input.type(), backgroundColor);
	background.copyTo(input(RoI));
	putText(input, text, pos, font, size, textColor, thickness, type, false);
}

void putTextBetter(Mat &input, string text, string position, Point offset, double size, Scalar textColor, Scalar backgroundColor, int thickness, int font, int type)
{
	transform(position.begin(), position.end(), position.begin(), ::toupper);	// zamiana na wielkie litery
	Size frameSize = input.size();
	int W = frameSize.width;
	int H = frameSize.height;
	Size textSize = getTextSize(text, font, size, thickness, 0);
	int h = textSize.height;
	int w = textSize.width;
	Point pos;
	if (position == "TL" || position == "TOPLEFT")
		pos = Point(0, h) + offset;
	else if (position == "TR" || position == "TOPRIGHT")
		pos = Point(W - w, h) + offset;
	else if (position == "BL" || position == "BOTTOMLEFT")
		pos = Point(0, H) + offset;
	else if (position == "BR" || position == "BOTTOMRIGHT")
		pos = Point(W - w, H) + offset;
	else if (position == "T" || position == "TOP")
		pos = Point((W - w) / 2, h) + offset;
	else if (position == "L" || position == "LEFT")
		pos = Point(0, (H + h) / 2) + offset;
	else if (position == "R" || position == "RIGHT")
		pos = Point(W - w, (H + h) / 2) + offset;
	else if (position == "B" || position == "BOTTOM")
		pos = Point((W - w) / 2, H) + offset;
	else if (position == "C" || position == "CENTER")
		pos = Point((W - w) / 2, (H + h) / 2) + offset;
	else
		pos = Point(0, h) + offset;
	Rect RoI = Rect(pos, pos + Point(w, -h));
	Mat background = Mat(RoI.height, RoI.width, input.type(), backgroundColor);
	background.copyTo(input(RoI));
	putText(input, text, pos, font, size, textColor, thickness, type, false);
}

string frame2time(int frameNum, double fps)
{
	int h, m, s;
	h = (int)(frameNum / (fps * 3600));
	m = (int)((frameNum / (fps * 60)) - (h * 3600));
	s = (int)((frameNum / fps) - (m * 60));
	char text_time[40];
	sprintf_s(text_time, "%02d:%02d:%02d", h, m, s);
	return(text_time);
}

bool overlapingContours_old(Mat img1, Mat img2, vector<Rect2d> &outputRectangles, double minArea)
{
	vector<vector<Point>> obj1, obj2; // wektor konturów na 1. i 2. obrazie
	findContours(img1, obj1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);	// znajdŸ kontury na 1. obrazie
	if (obj1.size() > 0)	// je¿eli znaleziono chocia¿ 1 kontur
	{
		sortContours(obj1, obj1, minArea);	// posortuj od najwiêkszych do najmniejszych
		findContours(img2, obj2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);	//znajdz kontury na 2. obrazie
	}
	else
		return false;
	if (obj2.size() == 0)
		return false;
	outputRectangles.clear();	// wyczyœæ wyjœciowy wektor prostok¹tów
	for (int i = 0; i < obj1.size(); i++)	// dla ka¿dego znalezionego konturu na obrazie 1.
	{
		Tracked object1(obj1[i]);	// przypisz kontur z 1. obrazu do zmiennej Tracked
		for (int j = 0; j < obj2.size(); j++)	// dla ka¿dego znalezionego konturu na obrazie 2.
		{
			if (pointPolygonTest(obj2[j], object1.centre(), false) >= 0)	// sprawdŸ czy kontur z obrazu 1. znajduje siê w którym kontórze z obrazu 2.
			{
				Tracked object2(obj2[j]);	// przypisz ten kontur z obrazu 2. do zmiennej Tracked
				outputRectangles.push_back(object2.bounding());	// dodaj jego prostok¹t do wektora wyjœciowego
			}
		}
	}
	return true;
}

bool overlapingContours(Mat img1, Mat img2, vector<Rect> &outputRectangles, double minArea)
{
	vector<vector<Point>> obj1, obj2; // wektor konturów na 1. i 2. obrazie
	findContours(img1, obj1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);	// znajdŸ kontury na 1. obrazie
	if (sortContours(obj1, obj1, minArea))	// je¿eli znaleziono chocia¿ 1 kontur > minArea
	{
		//sortContours(obj1, obj1, minArea);	// posortuj od najwiêkszych do najmniejszych
		findContours(img2, obj2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);	//znajdz kontury na 2. obrazie
	}
	else
		return false;
	if (obj2.size() == 0)	// je¿eli na drugim obrazie nie znaleziono konturów
		return false;		// zwróæ Fa³sz
	vector<Rect> tmp;		// tymczasowy wektor wyników
	//outputRectangles.clear();	// wyczyœæ wyjœciowy wektor prostok¹tów
	for (int i = 0; i < obj1.size(); i++)	// dla ka¿dego znalezionego konturu na obrazie 1.
	{
		Tracked object1(obj1[i]);	// przypisz kontur z 1. obrazu do zmiennej Tracked
		for (int j = 0; j < obj2.size(); j++)	// dla ka¿dego znalezionego konturu na obrazie 2.
		{
			if (pointPolygonTest(obj2[j], object1.centre(), false) >= 0)	// sprawdŸ czy kontur z obrazu 1. znajduje siê w którymœ konturze z obrazu 2.
			{
				Tracked object2(obj2[j]);	// przypisz ten kontur z obrazu 2. do zmiennej Tracked
				tmp.push_back(object2.bounding());	// dodaj jego prostok¹t do wektora wyjœciowego
			}
		}
	}
	if (tmp.size() > 0)	// je¿eli znaleziono chocia¿ 1 nak³adaj¹cy siê kontur
	{
		outputRectangles = tmp;	// przypisz wynik
		return true;			// zwróæ Prawda
	}
	else
		return true;	// zwróæ Fa³sz
}


Rect scaleRect(Rect input, double scale)
{
	assert(scale > 0);
	double ratio = scale - 1.0;
	Point offset((input.width * ratio) / 2, (input.height * ratio) / 2);
	Rect ans(input.x, input.y, input.width * scale, input.height * scale);
	ans -= offset;
	return (ans);
}

Rect scaleRect(Rect input, double xScale, double yScale)
{
	assert(xScale > 0);
	assert(yScale > 0);
	double xRatio = xScale - 1.0;
	double yRatio = yScale - 1.0;
	Point offset((input.width * xRatio) / 2, (input.height * yRatio) / 2);
	Rect ans(input.x, input.y, input.width * xScale, input.height * yScale);
	ans -= offset;
	return (ans);
}

Rect scaleRect(Rect input, Size srcImgSize, Size dstImgSize)
{
	double xScale = (double)dstImgSize.width / srcImgSize.width;
	double yScale = (double)dstImgSize.height / srcImgSize.height;
	Rect ans = input;
	ans.x = input.x * xScale;
	ans.y = input.y * yScale;
	ans.width = input.width * xScale;
	ans.height = input.height * yScale;
	return (ans);
}