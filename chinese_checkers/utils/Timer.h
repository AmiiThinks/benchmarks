//
//  Timer.h
//
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef Timer_H
#define Timer_H

#include <stdio.h>
#include <chrono>
#include <ctime>

class Timer
{
public:
	Timer();
	void StartTimer();
	double EndTimer();
	double GetElapsedTime();
private:
	bool ended;
	std::chrono::time_point<std::chrono::system_clock> start, end;
};

#endif /* defined(__Homework_3__Timer__) */
