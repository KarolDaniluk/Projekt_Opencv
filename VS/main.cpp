#include <iostream>
// SYSTEM
#include <chrono>
#include <string>
//#include <vector>
//#include <algorithm>
//#include <numeric>
// OPENCV - STANDARD
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// OPENCV - CUDA
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaobjdetect.hpp>
// OPENCV - EXTRA MODULES
#include <opencv2/tracking.hpp>
// W£ASNE
#include "utility.h"
#include "Tracked.h"
#include "UO.h"


using namespace std;
using namespace cv;
//using namespace cuda;
using namespace std::chrono;

bool update_morph_open = true;
bool update_morph_close = true;
bool update_gauss = true;

void callback_open(int pos, void* userdata)
{ update_morph_open = true; }

void callback_close(int pos, void* userdata)
{ update_morph_close = true; }

void callback_gauss(int pos, void *userdata)
{ update_gauss = true; }

void get_cuda_morph_filter(Ptr<cuda::Filter> &filter, int morphType, int kernel_size)
{
	kernel_size = max(1, kernel_size);
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(kernel_size, kernel_size));
	filter = cuda::createMorphologyFilter(morphType, CV_8U, kernel);
}

// Wersja zwyk³a
void Detect(string source)
{
	// CUDA
	cuda::setDevice(0);

	// Pozyskiwanie obrazu
	VideoCapture video(source);
	const double fps = video.get(CAP_PROP_FPS);
	const Size vidSize = Size((int)video.get(CAP_PROP_FRAME_WIDTH), (int)video.get(CAP_PROP_FRAME_HEIGHT));
	
	// Podstawowe zmienne
	int f = 0;			// licznik klatek
	char key = 0;		// wciœniêty klawisz
	bool paused = false;	// czy zapauzowaæ pracê

	// Obrazy CPU
	Mat frame,		// aktualna klatka obrazu
		gray,		// klatka w skali szaroœci
		background,	// t³o pozyskane z MOG2
		mog,		// ruch wykryty przez MOG2
		diff,		// ruch wykryty przez Diff
		heat,		// mapa ciep³a nieruchomych obiektów
		active,		// aktywacja po przekroczeniu progu
		mixed;		// po³¹czone obrazy metod MOG2 i Diff

	// Obrazy GPU
	cuda::GpuMat
		frameG,		// aktualna klatka obrazu
		grayG,		// klatka w skali szaroœci
		backgroundG,// t³o pozyskane z MOG2
		mogG,		// ruch wykryty przez MOG2
		diffG,		// ruch wykryty przez Diff
		heatG,		// mapa ciep³a nieruchomych obiektów
		activeG,	// aktywacja po przekroczeniu progu
		mixedG;		// po³¹czone obrazy metod MOG2 i Diff

	// Tworzenie okien
	const string wind1 = "Frame";
	const string wind2 = "Gray";
	const string wind3 = "Background";
	const string wind4 = "Movement MOG";
	const string wind5 = "Movement Diff";
	const string wind6 = "Heat";
	const string wind7 = "Active";
	const string wind8 = "Mixed";
	namedWindow(wind1, CV_WINDOW_AUTOSIZE);
	namedWindow(wind2, CV_WINDOW_AUTOSIZE);
	namedWindow(wind3, CV_WINDOW_AUTOSIZE);
	namedWindow(wind4, CV_WINDOW_AUTOSIZE);
	namedWindow(wind5, CV_WINDOW_AUTOSIZE);
	namedWindow(wind6, CV_WINDOW_AUTOSIZE);
	namedWindow(wind7, CV_WINDOW_AUTOSIZE);
	namedWindow(wind8, CV_WINDOW_AUTOSIZE);

	// Klasyfikatory kaskadowe
	//Ptr<cuda::CascadeClassifier> cascade_face_frontal = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_frontalface_alt.xml");
	//Ptr<cuda::CascadeClassifier> cascade_face_profile = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_profileface.xml");
	//Ptr<cuda::CascadeClassifier> cascade_fullbody = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_fullbody.xml");
	//Ptr<cuda::CascadeClassifier> cascade_eyes = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_eye_tree_eyeglasses.xml");

	// Klasyfikator HOG do wykrywania pieszych
	//Ptr<cuda::HOG> hog_pedestrians = cuda::HOG::create();		// tworzenie klasyfikatora
	//Mat detector = hog_pedestrians->getDefaultPeopleDetector();	// ³adowanie wbudowanego modelu
	//hog_pedestrians->setSVMDetector(detector);					// przypisanie modelu do klasyfikatora
	//detector.release();											// usuwanie za³adowanego modelu

	// Filtry cuda
	Ptr<cuda::Filter> filter_morph_open;	// otwarcie morfologiczne
	Ptr<cuda::Filter> filter_morph_close;	// zamkniêcie morfologiczne
	Ptr<cuda::Filter> filter_gauss;			// rozmycie gaussa
	
	// Zmienne suwaków
	int thresh_mog = 128;	// progowanie MOG
	int thresh_dif = 30;	// progowanie diff
	int alpha_tr = 20;		// szybkoœæ akumulacji w promilach
	int k_open = 2;			// rozmiar otwarcia morfologicznego
	int k_close = 7;		// rozmiar zamkniêcia morfologicznego
	int thresh_act = 250;	// punkt aktywacji mapy ciep³a
	int hist = 20;			// historia MOG - im mniejsze tym t³o szybciej siê adaptuje
	int speed = 0;		// prêdkoœæ odtwarzania (0 - mo¿liwie rzeczywista, 1 - najszybsza)
	int k_gauss = 3;	// rozmiar filtru gaussa
	int min_area = 10;	// minimalna powierzchnia nieruchomego obiektu

	// Agregacja (mapa ciep³a)
	double alpha;						// wspó³czynnik agregacji w zakresie od 0 do 1
	const int alpha_max = 1000;			// max wartoœæ alpha (alpha = alpha_tr / alpha_max);
	heat = Mat::zeros(vidSize, CV_32F);	// pocz¹tkowo mapa ciep³a to czarny obraz
	heatG.upload(heat);					// ³adowanie CPU -> GPU

	// Inne
	double mini, maxi;	// maksymalna i minimalna wartoœæ danego obrazu

	// Suwaki
	const string trackbars = "Trackbars";
	namedWindow(trackbars, CV_WINDOW_AUTOSIZE);
	resizeWindow(trackbars, Size(300, 120));
	createTrackbar("Speed", wind1, &speed, 1);
	createTrackbar("Gauss k", wind2, &k_gauss, 15, callback_gauss);
	createTrackbar("History", wind3, &hist, 100);
	setTrackbarMin("History", wind3, 1);
	createTrackbar("Thresh", wind4, &thresh_mog, 255);
	createTrackbar("Thresh", wind5, &thresh_dif, 255);
	createTrackbar("Alpha %%", wind6, &alpha_tr, 100);
	createTrackbar("Thresh", wind7, &thresh_act, 255);
	createTrackbar("Open k", trackbars, &k_open, 15, callback_open);
	createTrackbar("Close k", trackbars, &k_close, 15, callback_close);
	createTrackbar("Min area", trackbars, &min_area, 100);

	// Czas przed i po zakoñczeniu obliczeñ
	high_resolution_clock::time_point time1, time2;	// czas rozpoczêcia i zakoñczenia
	int duration = 0;	// czas obliczeñ w milisekundach (time2 - time1)

	// MOG2
	Ptr<cuda::BackgroundSubtractorMOG2> pMOG2 = cuda::createBackgroundSubtractorMOG2(2000);	// obiekt MOG

	while (key != 27)
	{
		if (!paused)
		{
			// czas rozpoczêcia obliczeñ
			time1 = high_resolution_clock::now();

			// Pozyskanie obrazu
			if (!video.read(frame)) break;	// przerwij pêtlê je¿eli klatka jest pusta
			frameG.upload(frame);	// ³adowanie klatki CPU -> GPU

			// Skala szaroœci
			cuda::cvtColor(frameG, grayG, COLOR_BGR2GRAY);	// skala szaroœci
			//cuda::equalizeHist(grayG, grayG);	// wyrównanie histogramu

			// Aktualizacja filtrów
			if (update_morph_open)
			{
				get_cuda_morph_filter(filter_morph_open, MORPH_OPEN, k_open);
				update_morph_open = !update_morph_open;
			}
			if (update_morph_close)
			{
				get_cuda_morph_filter(filter_morph_close, MORPH_CLOSE, k_close);
				update_morph_close = !update_morph_close;
			}
			if (update_gauss && k_gauss > 0)
			{
				restrictValue(k_gauss, 1, 15, 2);
				filter_gauss = cuda::createGaussianFilter(CV_8U, CV_8U, Size(k_gauss, k_gauss), 0, 0);
				update_gauss = !update_gauss;
			}

			//cuda::resize(grayG, grayG, Size(), 0.5, 0.5);	// zmniejsz rozmiar o 50%
			if (k_gauss > 0)
				filter_gauss->apply(grayG, grayG);

			// Wykrywanie ruchu MOG
			pMOG2->setHistory(hist * 100);	// d³ugoœæ historii - im mniejsze tym t³o szybciej siê adaptuje
			pMOG2->apply(grayG, mogG);		// wykrywanie ruchu Mog2
			pMOG2->getBackgroundImage(backgroundG);	// pozyskanie t³a
			if (thresh_mog > 0)
				cuda::threshold(mogG, mogG, thresh_mog, 255, THRESH_BINARY);	// progowanie binarne

			// Wykrywanie ruchu przez ró¿nicê jasnoœci (Diff)
			cuda::subtract(backgroundG, grayG, diffG);
			//cuda::absdiff(grayG, backgroundG, diffG, stream);
			cuda::threshold(diffG, mixedG, 16, 255, CV_THRESH_BINARY); // taki sam próg jaki ma MOG2 wewn¹trz
			if (thresh_dif > 0)
				cuda::threshold(diffG, diffG, thresh_dif, 255, THRESH_BINARY);	// progowanie binarne

			// £¹czenie obu metod
			cuda::add(mixedG, mogG, mixedG);

			// Otwarcie i zamkniêcie morfologiczne
			filter_morph_open->apply(mogG, mogG);		// otwarcie
			filter_morph_close->apply(mogG, mogG);		// zamkniêcie
			filter_morph_open->apply(diffG, diffG);		// otwarcie
			filter_morph_close->apply(diffG, diffG);	// zamkniêcie
			filter_morph_open->apply(mixedG, mixedG);	// otwarcie
			filter_morph_close->apply(mixedG, mixedG);	// zamkniêcie
													
			// Mapa ciep³a
			alpha = alpha_tr / (double)alpha_max;
			cuda_accumulateWeighted(diffG, heatG, alpha);		

			// Obraz aktywacji nieruchomych obiektów
			cuda::threshold(heatG, activeG, thresh_act, 255, CV_THRESH_BINARY);	// punkt aktywacji nieruchomych obiektów
			cuda::minMax(activeG, &mini, &maxi);	// odczyt minimalnej i maksymalnej wartoœci
			activeG.download(active);	// nie ma jeszcze findContours na CUDA :(
			active.convertTo(active, CV_8U);
		
			// Wykrywanie nieruchomych obiektów
			if (maxi == 255)
			{
				vector<Rect> objects;	// obiekty które nie poruszaj¹ siê od jakiegoœ czasu
				if (overlapingContours(active, mixed, objects, min_area))	// wyszukanie konturu na obrazie Mixed (po³¹czenie metod MOG2 i Diff)
				{
					for (int i = 0; i < objects.size(); i++)
						rectangle(frame, scaleRect(objects[i], 1.15), Scalar(200, 200, 200));
				}
			}

			// Pobieranie GPU -> CPU obrazów do wyœwietlenia
			grayG.download(gray);
			backgroundG.download(background);
			mogG.download(mog);
			diffG.download(diff);
			heatG.download(heat);
			heat.convertTo(heat, CV_8U);
			//activeG.download(active);
			mixedG.download(mixed);

			// Nak³adanie info na frame
			char text_calc[40];
			sprintf_s(text_calc, "Calc. time = %02d ms", duration); // czas obliczeñ 1 klatki
			putTextBetter(frame, text_calc, "TL", 0.5);
			putTextBetter(frame, frame2time(f, fps), "TR", 0.5);	// czas hh:mm:ss od pocz¹tku filmu

			// Nak³adanie info na heat
			cuda::minMax(heatG, &mini, &maxi);
			char heat_max[10];
			sprintf_s(heat_max, "Max = %.0f", maxi);
			putTextBetter(heat, heat_max, "TL", 0.5);

			// Wyœwietlanie
			imshow(wind1, frame);
			imshow(wind2, gray);
			imshow(wind3, background);
			imshow(wind4, mog);
			imshow(wind5, diff);
			imshow(wind6, heat);
			imshow(wind7, active);
			imshow(wind8, mixed);

			// Czas obliczeñ
			time2 = high_resolution_clock::now();
			duration = (int)duration_cast<milliseconds>(time2 - time1).count();
			f++;
		}
		// end of pause block

		// Odczyt klawisza
		int del;
		if (speed == 0)
			del = (int)max(1.0, 1000.0/fps - duration);	// pozosta³e opóŸnienie (po obliczeniach)
		else
			del = 1;
		key = waitKey(del);

		// Obs³uga klawiszy
		if (key == 32)	// pauza
			paused = !paused;
	}
	cout << "End of video. Press ENTER to exit" << endl;
	getchar();
	// Czyszczenie pamiêci
	video.release();			// Wyczysc obiekt przechwytywania video
	cv::destroyAllWindows();	// Zamknij wszystkie okna
}

// Wersja niezwyk³a?
void Detect2(string source)
{
	// CUDA
	cuda::setDevice(0);

	// Pozyskiwanie obrazu
	VideoCapture video(source);
	const double fps = video.get(CAP_PROP_FPS);
	const Size vidSize = Size((int)video.get(CAP_PROP_FRAME_WIDTH), (int)video.get(CAP_PROP_FRAME_HEIGHT));

	// Podstawowe zmienne
	int f = 0;			// licznik klatek
	char key = 0;		// wciœniêty klawisz
	bool paused = false;	// czy zapauzowaæ pracê

	// Obrazy CPU
	Mat frame,		// aktualna klatka obrazu
		gray,		// klatka w skali szaroœci
		background,	// t³o pozyskane z MOG2
		mog,		// ruch wykryty przez MOG2
		diff,		// ruch wykryty przez Diff
		heat,		// mapa ciep³a nieruchomych obiektów
		active,		// aktywacja po przekroczeniu progu
		mixed;		// po³¹czone obrazy metod MOG2 i Diff

	// Obrazy GPU
	cuda::GpuMat
		frameG,		// aktualna klatka obrazu
		grayG,		// klatka w skali szaroœci
		backgroundG,// t³o pozyskane z MOG2
		mogG,		// ruch wykryty przez MOG2
		diffG,		// ruch wykryty przez Diff
		heatG,		// mapa ciep³a nieruchomych obiektów
		activeG,	// aktywacja po przekroczeniu progu
		mixedG;		// po³¹czone obrazy metod MOG2 i Diff

	// Tworzenie okien
	const string wind1 = "Frame";
	const string wind2 = "Gray";
	const string wind3 = "Background";
	const string wind4 = "Movement MOG";
	const string wind5 = "Movement Diff";
	const string wind6 = "Heat";
	const string wind7 = "Active";
	const string wind8 = "Mixed";
	namedWindow(wind1, CV_WINDOW_AUTOSIZE);
	namedWindow(wind2, CV_WINDOW_AUTOSIZE);
	namedWindow(wind3, CV_WINDOW_AUTOSIZE);
	namedWindow(wind4, CV_WINDOW_AUTOSIZE);
	namedWindow(wind5, CV_WINDOW_AUTOSIZE);
	namedWindow(wind6, CV_WINDOW_AUTOSIZE);
	namedWindow(wind7, CV_WINDOW_AUTOSIZE);
	namedWindow(wind8, CV_WINDOW_AUTOSIZE);

	// Klasyfikatory kaskadowe
	//Ptr<cuda::CascadeClassifier> cascade_face_frontal = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_frontalface_alt.xml");
	//Ptr<cuda::CascadeClassifier> cascade_face_profile = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_profileface.xml");
	//Ptr<cuda::CascadeClassifier> cascade_fullbody = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_fullbody.xml");
	//Ptr<cuda::CascadeClassifier> cascade_eyes = cuda::CascadeClassifier::create("C:\\opencv_cuda_allcontrib_noworld\\etc\\haarcascades_cuda\\haarcascade_eye_tree_eyeglasses.xml");

	// Klasyfikator HOG do wykrywania pieszych
	//Ptr<cuda::HOG> hog_pedestrians = cuda::HOG::create();		// tworzenie klasyfikatora
	//Mat detector = hog_pedestrians->getDefaultPeopleDetector();	// ³adowanie wbudowanego modelu
	//hog_pedestrians->setSVMDetector(detector);					// przypisanie modelu do klasyfikatora
	//detector.release();											// usuwanie za³adowanego modelu

	// Filtry cuda
	Ptr<cuda::Filter> filter_morph_open;	// otwarcie morfologiczne
	Ptr<cuda::Filter> filter_morph_close;	// zamkniêcie morfologiczne
	Ptr<cuda::Filter> filter_gauss;			// rozmycie gaussa

	// Zmienne suwaków
	int thresh_mog = 128;	// progowanie MOG
	int thresh_dif = 30;	// progowanie diff
	int alpha_tr = 20;		// szybkoœæ akumulacji w promilach
	int k_open = 2;			// rozmiar otwarcia morfologicznego
	int k_close = 7;		// rozmiar zamkniêcia morfologicznego
	int thresh_act = 250;	// punkt aktywacji mapy ciep³a
	int hist = 20;			// historia MOG - im mniejsze tym t³o szybciej siê adaptuje
	int speed = 1;		// prêdkoœæ odtwarzania (0 - mo¿liwie rzeczywista, 1 - najszybsza)
	int k_gauss = 3;	// rozmiar filtru gaussa
	int min_area = 1;	// minimalna powierzchnia nieruchomego obiektu

	// Agregacja (mapa ciep³a)
	double alpha;						// wspó³czynnik agregacji w zakresie od 0 do 1
	const int alpha_max = 1000;			// max wartoœæ alpha (alpha = alpha_tr / alpha_max);
	heat = Mat::zeros(vidSize, CV_32F);	// pocz¹tkowo mapa ciep³a to czarny obraz
	heatG.upload(heat);					// ³adowanie CPU -> GPU

	// Inne
	double mini, maxi;	// maksymalna i minimalna wartoœæ danego obrazu

	// Suwaki
	const string trackbars = "Trackbars";
	namedWindow(trackbars, CV_WINDOW_AUTOSIZE);
	resizeWindow(trackbars, Size(300, 120));
	createTrackbar("Speed", wind1, &speed, 1);
	createTrackbar("Gauss k", wind2, &k_gauss, 15, callback_gauss);
	createTrackbar("History", wind3, &hist, 100);
	setTrackbarMin("History", wind3, 1);
	createTrackbar("Thresh", wind4, &thresh_mog, 255);
	createTrackbar("Thresh", wind5, &thresh_dif, 255);
	createTrackbar("Alpha %%", wind6, &alpha_tr, 100);
	createTrackbar("Thresh", wind7, &thresh_act, 255);
	createTrackbar("Open k", trackbars, &k_open, 15, callback_open);
	createTrackbar("Close k", trackbars, &k_close, 15, callback_close);
	createTrackbar("Min area", trackbars, &min_area, 100);

	// Czas przed i po zakoñczeniu obliczeñ
	high_resolution_clock::time_point time1, time2;	// czas rozpoczêcia i zakoñczenia
	int duration = 0;	// czas obliczeñ w milisekundach (time2 - time1)

	// MOG2
	Ptr<cuda::BackgroundSubtractorMOG2> pMOG2 = cuda::createBackgroundSubtractorMOG2(2000);	// obiekt MOG

	UO plecak;

	while (key != 27)
	{
		if (!paused)
		{
			// czas rozpoczêcia obliczeñ
			time1 = high_resolution_clock::now();

			// Pozyskanie obrazu
			if (!video.read(frame)) break;	// przerwij pêtlê je¿eli klatka jest pusta
			frameG.upload(frame);	// ³adowanie klatki CPU -> GPU

			// Skala szaroœci
			cuda::cvtColor(frameG, grayG, COLOR_BGR2GRAY);	// skala szaroœci
			//cuda::equalizeHist(grayG, grayG);	// wyrównanie histogramu

			// Aktualizacja filtrów
			if (update_morph_open)
			{
				get_cuda_morph_filter(filter_morph_open, MORPH_OPEN, k_open);
				update_morph_open = !update_morph_open;
			}
			if (update_morph_close)
			{
				get_cuda_morph_filter(filter_morph_close, MORPH_CLOSE, k_close);
				update_morph_close = !update_morph_close;
			}
			if (update_gauss && k_gauss > 0)
			{
				restrictValue(k_gauss, 1, 15, 2);
				filter_gauss = cuda::createGaussianFilter(CV_8U, CV_8U, Size(k_gauss, k_gauss), 0, 0);
				update_gauss = !update_gauss;
			}

			//cuda::resize(grayG, grayG, Size(), 0.5, 0.5);	// zmniejsz rozmiar o 50%
			if (k_gauss > 0)
				filter_gauss->apply(grayG, grayG);

			// Wykrywanie ruchu MOG
			pMOG2->setHistory(hist * 100);	// d³ugoœæ historii - im mniejsze tym t³o szybciej siê adaptuje
			pMOG2->apply(grayG, mogG);		// wykrywanie ruchu Mog2
			pMOG2->getBackgroundImage(backgroundG);	// pozyskanie t³a
			if (thresh_mog > 0)
				cuda::threshold(mogG, mogG, thresh_mog, 255, THRESH_BINARY);	// progowanie binarne

			// Wykrywanie ruchu przez ró¿nicê jasnoœci (Diff)
			cuda::subtract(backgroundG, grayG, diffG);
			//cuda::absdiff(grayG, backgroundG, diffG, stream);
			cuda::threshold(diffG, mixedG, 16, 255, CV_THRESH_BINARY); // taki sam próg jaki ma MOG2 wewn¹trz
			if (thresh_dif > 0)
				cuda::threshold(diffG, diffG, thresh_dif, 255, THRESH_BINARY);	// progowanie binarne

			// £¹czenie obu metod
			cuda::add(mixedG, mogG, mixedG);

			// Otwarcie i zamkniêcie morfologiczne
			filter_morph_open->apply(mogG, mogG);		// otwarcie
			filter_morph_close->apply(mogG, mogG);		// zamkniêcie
			filter_morph_open->apply(diffG, diffG);		// otwarcie
			filter_morph_close->apply(diffG, diffG);	// zamkniêcie
			filter_morph_open->apply(mixedG, mixedG);	// otwarcie
			filter_morph_close->apply(mixedG, mixedG);	// zamkniêcie

			// Mapa ciep³a
			alpha = alpha_tr / (double)alpha_max;
			cuda_accumulateWeighted(diffG, heatG, alpha);

			// Obraz aktywacji nieruchomych obiektów
			cuda::threshold(heatG, activeG, thresh_act, 255, CV_THRESH_BINARY);	// punkt aktywacji nieruchomych obiektów
			cuda::minMax(activeG, &mini, &maxi);	// odczyt minimalnej i maksymalnej wartoœci
			activeG.download(active);	// nie ma jeszcze findContours na CUDA :(
			active.convertTo(active, CV_8U);

			// Wykrywanie nieruchomych obiektów
			//grayG.download(gray);		// w celu wykrywania features dostarczamy szary obraz
			vector<Rect> objects;		// pokrywaj¹ce siê prostok¹ty
			if (maxi == 255)
				overlapingContours(active, mixed, objects, min_area);
			plecak.update(objects, frame, true);
			if (plecak.initialized)
				rectangle(frame, plecak.rect, plecak.color(), plecak.thickness());

			// Pobieranie GPU -> CPU obrazów do wyœwietlenia
			grayG.download(gray);
			backgroundG.download(background);
			mogG.download(mog);
			diffG.download(diff);
			heatG.download(heat);
			heat.convertTo(heat, CV_8U);
			//activeG.download(active);
			mixedG.download(mixed);

			// Nak³adanie info na frame
			char text_calc[40];
			sprintf_s(text_calc, "Calc. time = %02d ms", duration); // czas obliczeñ jednej klatki
			putTextBetter(frame, text_calc, "TL", 0.5);
			putTextBetter(frame, frame2time(f, fps), "TR", 0.5);	// czas hh:mm:ss od pocz¹tku filmu

			// Nak³adanie info na heat
			cuda::minMax(heatG, &mini, &maxi);
			char heat_max[10];
			sprintf_s(heat_max, "Max = %.0f", maxi);
			putTextBetter(heat, heat_max, "TL", 0.5);

			// Wyœwietlanie
			imshow(wind1, frame);
			imshow(wind2, gray);
			imshow(wind3, background);
			imshow(wind4, mog);
			imshow(wind5, diff);
			imshow(wind6, heat);
			imshow(wind7, active);
			imshow(wind8, mixed);

			// Czas obliczeñ
			time2 = high_resolution_clock::now();
			duration = (int)duration_cast<milliseconds>(time2 - time1).count();
			f++;
		}
		// end of pause block

		// Odczyt klawisza
		int del;
		if (speed == 0)
			del = (int)max(1.0, 1000.0 / fps - duration);	// pozosta³e opóŸnienie (po obliczeniach)
		else
			del = 1;
		key = waitKey(del);

		// Obs³uga klawiszy
		if (key == 32)	// pauza
			paused = !paused;
	}
	cout << "End of video. Press ENTER to exit" << endl;
	getchar();
	// Czyszczenie pamiêci
	video.release();			// Wyczyœæ obiekt przechwytywania video
	cv::destroyAllWindows();	// Zamknij wszystkie okna
}


int main()
{
	cuda::printShortCudaDeviceInfo(0);
	string source1 = "PETS_2006\\A\\Cam1_mini_short.mp4";
	string source2 = "PETS_2006\\A\\Cam2_mini.avi";
	Detect2(source1);
	// Exit
	//cout << "Press ENTER to exit";
	//getchar();
	return(0);
}
