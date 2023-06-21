//
//  ThreadedSolve.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 7/23/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include "ThreadedSolver.h"
#include <vector>
#include <stdio.h>
#include <stdint.h>
#include "CCheckers.h"

std::vector<bool> theWins;
int proofPlayer = 0;
int numThreads = 4;
bool proven = false;
void *SolveThread(void *whichRange);

void SolveWithThreads(int threadCount)
{
	numThreads = threadCount;
	CCheckers cc;
	CCState s;
	
	cc.Reset(s);
	//uint64_t root = cc.rank(s);
	uint64_t maxRank = cc.getMaxRank();
	theWins.resize(maxRank);
	
	printf("Finding won positions for player %d\n", proofPlayer);
	printf("%lld total states\n", maxRank);

	std::vector<pthread_t> threads;
	
	for (int x = 0; x < numThreads; x++)
	{
		pthread_t thread;
		pthread_create(&thread, NULL, SolveThread, (void *)x);
		threads.push_back(thread);
	}
	for (unsigned int x = 0; x < threads.size(); x++)
		pthread_join(threads[x], 0);

	//	
	//	printf("%lld states unranked; %lld were winning; %lld were tech. illegal\n",
	//		   legalStates, winningStates, illegalStates);
	//	int changed = 0;
	//	int round = 0;
	//	printf("Win not proven\n");
}


void *SolveThread(void *whichRange)
{
	CCheckers cc;
	CCState s;
	cc.Reset(s);
	uint64_t root = cc.rank(s);
	uint64_t maxRank = cc.getMaxRank();
	
	uint64_t range = maxRank/numThreads;
	uint64_t start, end;
	uint64_t which = (long)whichRange;
	start = range*which;//0;//
	end = range*(which+1);//maxRank;//
	printf("Hello World! %llu Solving [%lld to %lld)\n", which, start, end);
	
	//	uint64_t winningStates = 0;
	//	uint64_t illegalStates = 0;
	//	uint64_t legalStates = 0;
	
	int changed = 0;
	int round = 0;
	do {
		changed = 0;
		// propagating wins
		if (theWins[root])
		{
			printf("Win proven for player %d\n", proofPlayer);
			proven = true;
			pthread_exit(NULL);
			//exit(0);
			//return;
		}
		if (proven) pthread_exit(NULL);
		
		for (uint64_t val = start; val < end; val++)
		{
			//			if (val%numThreads != which)
			//				continue;
			if (theWins[val])
			{
				continue;
			}
			else if (cc.unrank(val, s))
			{
				if (cc.Winner(s) == (1-proofPlayer)) // can't prove their win a loss
					continue;
				if (cc.Done(s))
				{
					if (cc.Winner(s) == proofPlayer && s.toMove == 1-proofPlayer) // only when we have a bad ranking function
					{
						changed++;
						theWins[val] = true;
						continue;
					}
				}
				CCMove *m = cc.getMoves(s);
				bool done = false;
				bool proverToMove = (s.toMove==proofPlayer);
				for (CCMove *t = m; t && !done; t = t->next)
				{
					cc.ApplyMove(s, t);
					uint64_t succ = cc.rank(s);
					if ((proverToMove) && theWins[succ])
					{
						theWins[val] = true;
						done = true;
						changed++;
					}
					if ((!proverToMove) && (!theWins[succ]))
					{
						done = true;
					}
					cc.UndoMove(s, t);
				}
				if (!done && !proverToMove)
				{
					theWins[val] = true;
					changed++;
				}
				cc.freeMove(m);
			}
		}
		printf("[%llu] round %d; %d changed\n", which, round++, changed);
	} while (true);
	
	
	pthread_exit(NULL);
}
