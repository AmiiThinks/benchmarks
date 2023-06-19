//
//  RankingTests.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/25/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include <cassert>
#include "RankingTests.h"

void TestRankingSpeed()
{
	Timer t;
	CCheckers cc;
	CCState s;
	CCDefaultRank d;
	CCLocalRank12 l;
	
	t.StartTimer();
	for (uint64_t r = 0; r < d.getMaxRank(); r++)
	{
		d.unrank(r, s);
	}
	t.EndTimer();
	printf("Default unrank: %1.2fs elapsed\n", t.GetElapsedTime());
	
	t.StartTimer();
	for (uint64_t r = 0; r < l.getMaxRank(); r++)
	{
		l.unrank(r, s);
	}
	t.EndTimer();
	printf("Local unrank: %1.2fs elapsed\n", t.GetElapsedTime());
}

