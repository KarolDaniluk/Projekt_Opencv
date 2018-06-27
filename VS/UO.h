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

	Rect rect;		// prostok¹t - kszta³t obiektu
	time_t time_init, time_active, time_alarm;	// czas rozpoczêcia: inicializacji, aktywacji, alarmu
	int active_time = 3;	// (sec) czas aktywacji obiektu od jego inicializacji
	int alarm_time = 5;		// (sec) czas wywo³ania alarmu od jego aktywacji
	Mat frame_last, frame_init, frame_active, frame_alarm;	// klatka zapisana przy rozpoczêciu: inicializacji, aktywacji, alarmu
	bool initialized = false;	// czy obiekt zosta³ zainicjalizowany
	bool activated = false;	// czy obiekt zosta³ aktywowany
	bool alarmed = false;	// czy zosta³ w³¹czony alarm
	bool moving = false;	// czy obiekt siê porusza
	//Point pos_init, pos_active, pos_alarm;	// pozycja (TL) obiektu w momencie: inicializacji, aktywacji, alarmu
	double distTracked = 0;	// odleg³oœæ jak¹ pokona³ obiekt w ca³ym swoim ¿yciu
	bool print = true;
	
	// Czy dostarczaæ do obiektu obraz konturów lub kontur, tak aby obiekt oblicza³
	// przemieszczenie na podstawie œrodka ciê¿koœci konturu, zamiast œrodka prostok¹ta?

	//bool empty();	// czy obiekt jest pusty /// czy to w ogóle potrzebne skoro jest: bool initialized?
	void init(Rect rectangle, Mat frame);	// rozpocznij obiekt
	void activate(Mat frame);	// aktywuj obiekt
	void alarm(Mat frame);		// w³¹cz alarm
	void update(vector<Rect> rectangles, Mat frame, bool releaseIfEmpty);	// aktualizuj obiekt - wykonaj wyszstkie czynnoœci odpowiednie dla aktualnego stanu
	Rect overlaping(vector<Rect> rectangles);
	void release();
	int initTime();		// czas (sec) trwania od inizjalizacji
	int activeTime();	// czas (sec) trwania aktywnoœci
	int alarmTime();	// czas (sec) trwania alarmu
	Scalar color();		// zwróæ kolor w zale¿noœci od stanu
	int thickness();	// zwróæ gruboœæ linii w zale¿noœci od stanu
	Point position();	// po³o¿enie (TL) obiektu
	Point center();		// po³o¿enie œrodka obiektu
	Point center(Rect rectangle);		// po³o¿enie œrodka prostok¹ta
	int get_state();			// zwróæ stan: 0-pusty, 1 - zainicjalizowany, 2 - aktywny, 3 - alarm
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

