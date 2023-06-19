//
//  BackwardSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 4/22/16.
//  Copyright © 2016 NS Software. All rights reserved.
//

#ifndef BackwardSolver_h
#define BackwardSolver_h

#include <stdint.h>
#include <vector>
#include <thread>
#include "SolveStats.h"
#include "Timer.h"
#include "CCheckers.h"
#include "SharedQueue.h"

template <bool expandAll = true, bool winOnly = true, bool immediateWin = false>
class BackwardSolver
{
public:
	void SolveGame(std::vector<int8_t> &wins, bool concurrent = true);
private:
	void SetProvenStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
						 int numThreads, int provingPlayer, stats &stat);
	
	void ProveStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
					 int numThreads, int provingPlayer, stats &stat, int round);
	
};



template <bool expandAll, bool winOnly, bool immediateWin>
void BackwardSolver<expandAll, winOnly, immediateWin>::SetProvenStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
																 int numThreads, int provingPlayer, stats &stat)
{
	CCheckers cc;
	CCState s;
	stats local;
	for (uint64_t val = threadID; val < maxRank; val+=numThreads)
	{
		result[val] = 0;
		if (cc.unrank(val, s))
		{
			local.legalStates++;
			if (cc.Done(s))
			{
				if (cc.Winner(s) == provingPlayer)
				{
					if (s.toMove == 1-provingPlayer)
					{
						result[val] = 1;
						local.winningStates++;
					}
					else {
						local.illegalStates++;
					}
				}
				if (cc.Winner(s) == 1-provingPlayer)
				{
					if (s.toMove == provingPlayer)
					{
						result[val] = -1;
						local.losingStates++;
					}
					else {
						local.illegalStates++;
					}
				}
				
			}
		}
		else {
		}
	}
	stat += local;
}

template <bool expandAll, bool winOnly, bool immediateWin>
void BackwardSolver<expandAll, winOnly, immediateWin>::ProveStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
															 int numThreads, int provingPlayer, stats &stat, int round)
{
	CCheckers cc;
	CCState s;
	stats local;
	
	for (uint64_t val = 0; val < maxRank; val++)
	{
		//if ((expandAll && result[val] > 0) || (!expandAll && result[val] == round+1))
		//if (result[val] == round+1) // proven win
		if ((expandAll && (result[val] != 0) && !winOnly) ||
			(expandAll && (result[val] > 0) && winOnly) ||
			(!expandAll && (abs(result[val]) == round+1) && !winOnly) ||
			(!expandAll && (result[val] == round+1) && winOnly))
		{
			cc.unrank(val, s);
			local.unrank++;
			
			CCMove *backMoves = cc.getReverseMoves(s);
			local.backwardChildren += backMoves->length();
			local.backwardExpansions++;
			for (CCMove *t = backMoves; t; t = t->next)
			{
				cc.UndoMove(s, t);
				local.undo++;
				
				uint64_t parentRank = cc.rank(s);
				local.rank++;

				if ((parentRank%numThreads) == threadID && result[parentRank] == 0)// && cc.Winner(s) == -1)//1-provingPlayer)
				{
					bool earlyTerminate = false;
					bool allProven = true;
					bool proverToMove = (s.toMove==provingPlayer);
					
					if (proverToMove && immediateWin && result[val] > 0) // immediate proof max nodes from winning child
					{
						result[parentRank] = round+2;
						local.changed++;
						local.winningStates++;
					}
					else if (!proverToMove && immediateWin && result[val] < 0) // immediate proof min nodes from losing child
					{
						result[parentRank] = -(round+2);
						local.changed++;
						local.losingStates++;
					}
					else {
						CCMove *forwardMoves = cc.getMoves(s);
						local.forwardChildren += forwardMoves->length();
						local.forwardExpansions++;
						
						for (CCMove *next = forwardMoves; next && !earlyTerminate; next = next->next)
						{
							cc.ApplyMove(s, next);
							local.apply++;
							uint64_t succ = cc.rank(s);
							local.undo++;
							cc.UndoMove(s, next);

							if (proverToMove && result[succ] > 0 && (expandAll || result[succ] <= round+1))
							{
								allProven = false;
								earlyTerminate = true;

								result[parentRank] = round+2;
								local.changed++;
								local.winningStates++;
								break;
							} // Applied Computing
							else if (!winOnly && (!proverToMove) && result[succ] < 0 && (expandAll || result[succ] >= -(round+1)))
							{
								allProven = false;
								earlyTerminate = true;
								
								result[parentRank] = -(round+2);
								local.changed++;
								local.losingStates++;
								break;
							}
							else if (result[succ] == 0 || (winOnly && result[succ] < 0) || (!expandAll && abs(result[succ]) >= round+2))
							{
								allProven = false;
							}
						}
						
						if (earlyTerminate == false && allProven == true && !proverToMove)
						{
							result[parentRank] = round+2;
							local.winningStates++;
							local.changed++;
						}
						else if (earlyTerminate == false && allProven == true && proverToMove && !winOnly)
						{
							result[parentRank] = -(round+2);
							local.losingStates++;
							local.changed++;
						}
						
						cc.freeMove(forwardMoves);
					}
				}
				
				local.apply++;
				cc.ApplyMove(s, t);
			}
			cc.freeMove(backMoves);
		}
	}
	
	stat += local;
}



template <bool expandAll, bool winOnly, bool immediateWin>
void BackwardSolver<expandAll, winOnly, immediateWin>::SolveGame(std::vector<int8_t> &wins, bool concurrent)
{
	printf("--== Beginning backward solve ==--\n");
	printf("Expand all: %s; win only: %s; immedate wins: %s\n", expandAll?"true":"false", winOnly?"true":"false", immediateWin?"true":"false");
	int threadCnt = std::thread::hardware_concurrency();
	if (!concurrent)
		threadCnt = 1;
	printf("Running with %d threads\n", threadCnt);
	int provingPlayer = 0;
	CCheckers cc;
	CCState s;
	Timer total;
	total.StartTimer();
	
	cc.Reset(s);
	uint64_t root = cc.rank(s);
	uint64_t maxRank = cc.getMaxRank();
	wins.clear();
	wins.resize(maxRank);
	
	stats stat;
	
	printf("Finding all terminal positions\n");
	
	std::vector<std::thread *> threads(threadCnt);
	
	Timer t;
	t.StartTimer();
	for (int x = 0; x < threads.size(); x++)
	{
		threads[x] = new std::thread(&BackwardSolver<expandAll, winOnly, immediateWin>::SetProvenStates, this, std::ref(wins), maxRank, x, threads.size(),
									 provingPlayer, std::ref(stat));
	}
	for (int x = 0; x < threads.size(); x++)
	{
		threads[x]->join();
		delete threads[x];
		threads[x] = 0;
	}
	t.EndTimer();
	printf("%1.3fs elapsed\n", t.GetElapsedTime());
	std::cout << maxRank << " of " << maxRank << " 100% complete " <<
	stat.legalStates << " legal " << stat.winningStates << " winning " << stat.losingStates << " losing " << stat.illegalStates << " illegal" << std::endl;
	
	//	printf("%lld states unranked; %lld were winning; %lld were tech. illegal\n",
	//		   stat.legalStates, stat.winningStates, stat.illegalStates);
	
	stat.changed = 0;
	int round = 0;
	do {
		stat.changed = 0;
		t.StartTimer();
		// propagating wins
		for (int x = 0; x < threads.size(); x++)
		{
			threads[x] = new std::thread(&BackwardSolver<expandAll, winOnly, immediateWin>::ProveStates, this, std::ref(wins), maxRank, x, threads.size(),
										 provingPlayer, std::ref(stat), round);
		}
		for (int x = 0; x < threads.size(); x++)
		{
			threads[x]->join();
			delete threads[x];
			threads[x] = 0;
		}
		t.EndTimer();
		
		printf("round %d; %llu changed; expansions: %llu (f) %llu (b); %llu of %llu proven (%llu wins %llu losses); %1.3fs elapsed\n", round++,
			   stat.changed, stat.forwardExpansions, stat.backwardExpansions, stat.winningStates+stat.losingStates, maxRank, stat.winningStates, stat.losingStates, t.GetElapsedTime());
	} while (stat.changed != 0);

	std::cout << "ExpandAll: " << expandAll << " WinOnly " << winOnly << " immediateWin " << immediateWin << "\n";
	std::cout << stat << "\n";
	
	total.EndTimer();
	printf("%1.3fs elapsed\n", total.GetElapsedTime());
	
	if (wins[root])
	{
		printf("Win proven for player %d\n", provingPlayer);
	}
	
	
	uint64_t w=0,l=0,d=0;
	for (uint64_t x = 0; x < wins.size(); x++)
	{
		if (wins[x] > 0)
			w++;
		else if (wins[x] < 0)
			l++;
		else if (wins[x] == 0)
		{
			d++;
//			cc.unrank(x, s);
//			s.Print();
//			printf("State %llu unproven\n", x);
		}
	}
	printf("%llu wins, %llu losses; %llu draws\n", w, l, d);
}

#endif /* BackwardSolver_h */
