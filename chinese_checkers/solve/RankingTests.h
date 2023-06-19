//
//  RankingTests.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/25/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef RankingTests_h
#define RankingTests_h

#include "Timer.h"
#include "CCheckers.h"
#include "CCRankings.h"

void TestRankingSpeed();

template <typename T>
void TestChildLocality()
{
	CCheckers cc;
	CCState s;
	T l;
	uint64_t maxDiff = 0;
	for (uint64_t r = 0; r < l.getMaxRank(); r++)
	{
		l.unrank(r, s);
		uint64_t localMax = 0, localMin = l.getMaxRank();
		CCMove *m = cc.getMoves(s);
		for (CCMove *mm = m; mm; mm = mm->next)
		{
			cc.ApplyMove(s, mm);
			uint64_t r2 = l.rank(s);
			localMax = std::max(localMax, r2);
			localMin = std::min(localMin, r2);
			cc.UndoMove(s, mm);
			maxDiff = std::max(maxDiff, localMax-localMin);
		}
		cc.freeMove(m);
	}
	printf("With %s, maxdiff %llu with %llu values\n", l.name(), maxDiff, l.getMaxRank());
}

template <typename T>
void TestRankingCorrectness()
{
	printf("Testing correctness after forward/backwards actions\n");
	Timer t;
	CCheckers cc;
	CCState s, s2;
	T l;
	
	t.StartTimer();
	for (uint64_t r = 0; r < l.getMaxRank(); r++)
	{
		l.unrank(r, s);
		s.Verify();
		CCMove *m = cc.getMoves(s);
		for (CCMove *mm = m; mm; mm = mm->next)
		{
			cc.ApplyMove(s, mm);
			uint64_t r2 = l.rank(s);
			l.unrank(r2, s2);
			s2.Verify();
			if (s != s2)
			{
				printf("Rank followed by unranking failure\n");
				s.PrintASCII();
				s2.PrintASCII();
				exit(0);
			}
			cc.UndoMove(s, mm);
		}
		cc.freeMove(m);
		
		m = cc.getReverseMoves(s);
		for (CCMove *mm = m; mm; mm = mm->next)
		{
			cc.ApplyReverseMove(s, mm);
			uint64_t r2 = l.rank(s);
			l.unrank(r2, s2);
			s2.Verify();
			if (s != s2)
			{
				printf("Rank followed by unranking failure\n");
				s.PrintASCII();
				s2.PrintASCII();
				exit(0);
			}
			cc.UndoReverseMove(s, mm);
		}
		cc.freeMove(m);
		
		if (l.rank(s) != r)
		{
			printf("Unrank followed by ranking failure\n");
		}
	}
	t.EndTimer();
	
	printf("Local unrank: %1.2fs elapsed\n", t.GetElapsedTime());
}

template <typename T1, typename T2>
void TestMapping()
{
	Timer t;
	T1 d;
	T2 l;
	std::vector<bool> hit;
	hit.resize(d.getMaxRank());
	CCheckers cc;
	CCState s, s2;
	t.StartTimer();
	for (uint64_t r = 0; r < d.getMaxRank(); r++)
	{
		d.unrank(r, s);
		//s.Print();
		int64_t v = l.rank(s);
		//printf("Got rank %llu -> %llu (of %llu)\n", r, v, l.getMaxRank());
		if (hit[v])
		{
			printf("Double hit index %llu\n", r);
		}
		hit[v] = true;
		
		l.unrank(v, s2);
		if (s != s2)
		{
			printf("States not matching:\n");
			s.PrintASCII();
			s2.PrintASCII();
		}
		assert(s == s2);
	}
	t.EndTimer();
	printf("Done testing ranking; %1.2fs elapsed\n", t.GetElapsedTime());
	exit(0);
}


#endif /* RankingTests_h */
