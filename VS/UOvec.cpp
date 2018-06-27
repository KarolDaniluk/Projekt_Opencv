#include "UOvec.h"



UOvec::UOvec()
{
}


UOvec::~UOvec()
{
}

void UOvec::add(vector<Rect> newObjects)
{
	vector<Rect> buffer = objects;
	for (int i = 0; i < newObjects.size(); i++)
	{
		for (int j = 0; j < objects.size(); j++)
		{
			if ((newObjects[i] & objects[j]).area() == 0)
				buffer.push_back(newObjects[i]);
		}
	}
	this->objects = buffer;
}