#pragma once
#include <iostream>
#include <ctime>
#include <numeric>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class UO
{
public:
	UO();
	~UO();

	Rect rect;		// prostok�t - kszta�t obiektu
	time_t time_init, time_active, time_alarm;	// czas rozpocz�cia: inicializacji, aktywacji, alarmu
	int active_time = 3;	// (sec) czas aktywacji obiektu od jego inicializacji
	int alarm_time = 5;		// (sec) czas wywo�ania alarmu od jego aktywacji
	Mat frame_last, frame_init, frame_active, frame_alarm;	// klatka zapisana przy rozpocz�ciu: inicializacji, aktywacji, alarmu
	bool initialized = false;	// czy obiekt zosta� zainicjalizowany
	bool activated = false;	// czy obiekt zosta� aktywowany
	bool alarmed = false;	// czy zosta� w��czony alarm
	bool moving = false;	// czy obiekt si� porusza
	//Point pos_init, pos_active, pos_alarm;	// pozycja (TL) obiektu w momencie: inicializacji, aktywacji, alarmu
	double distTracked = 0;	// odleg�o�� jak� pokona� obiekt w ca�ym swoim �yciu
	bool print = true;
	
	// Czy dostarcza� do obiektu obraz kontur�w lub kontur, tak aby obiekt oblicza�
	// przemieszczenie na podstawie �rodka ci�ko�ci konturu, zamiast �rodka prostok�ta?

	//bool empty();	// czy obiekt jest pusty /// czy to w og�le potrzebne skoro jest: bool initialized?
	void init(Rect rectangle, Mat frame);	// rozpocznij obiekt
	void activate(Mat frame);	// aktywuj obiekt
	void alarm(Mat frame);		// w��cz alarm
	void update(vector<Rect> rectangles, Mat frame, bool releaseIfEmpty);	// aktualizuj obiekt - wykonaj wyszstkie czynno�ci odpowiednie dla aktualnego stanu
	Rect overlaping(vector<Rect> rectangles);
	void release();
	int initTime();		// czas (sec) trwania od inizjalizacji
	int activeTime();	// czas (sec) trwania aktywno�ci
	int alarmTime();	// czas (sec) trwania alarmu
	Scalar color();		// zwr�� kolor w zale�no�ci od stanu
	int thickness();	// zwr�� grubo�� linii w zale�no�ci od stanu
	Point position();	// po�o�enie (TL) obiektu
	Point center();		// po�o�enie �rodka obiektu
	Point center(Rect rectangle);		// po�o�enie �rodka prostok�ta
	int get_state();			// zwr�� stan: 0-pusty, 1 - zainicjalizowany, 2 - aktywny, 3 - alarm
	vector<Rect> humans;

private:
	Ptr<cuda::HOG> detector;
	void checkState(Rect rectangle, Mat frame);
	double calcDist(Rect rectangle);
	double calcDist(Rect rectangle1, Rect rectangle2);
	bool detectHumans(Mat frame, int expandScale = 2);
	void getHOGdetector();
	double humanDist();
};

