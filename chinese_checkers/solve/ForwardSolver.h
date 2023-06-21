//
//  SimpleSolve.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 7/23/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include <stdint.h>
#include <vector>
#include "CCheckers.h"
#include "SolveStats.h"
#include "Timer.h"
#include <thread>

template <bool strictDepth, bool winOnly, bool backProp>
class ForwardSolver
{
public:
	void SolveGame(std::vector<int8_t> &wins, bool threaded = true);
private:
	void SetProvenStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
						 int numThreads, int provingPlayer, stats &stat);
	void ProveStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
					 int numThreads, int provingPlayer, stats &stat, int round);

};


template <bool strictDepth, bool winOnly, bool backProp>
void ForwardSolver<strictDepth, winOnly, backProp>::SetProvenStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
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
			int winner = cc.Winner(s);
			if (winner == provingPlayer)
			{
				result[val] = 1;
				local.winningStates++;
			}
			else if (winner == 1-provingPlayer)
			{
				result[val] = -1;
				local.losingStates++;
			}
		}
	}
	stat += local;
}

template <bool strictDepth, bool winOnly, bool backProp>
void ForwardSolver<strictDepth, winOnly, backProp>::ProveStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
																			 int numThreads, int provingPlayer, stats &stat, int round)
{
	CCheckers cc;
	CCState s;
	stats local;
	
	for (uint64_t val = threadID; val < maxRank; val+=numThreads)
	{
		if (result[val] != 0) // already proven
			continue;
		
		local.unrank++;
		if (cc.unrank(val, s))
		{
			bool proven = true;
			bool proverToMove = (s.toMove==provingPlayer);
			
			CCMove *m = cc.getMoves(s);
			local.forwardChildren += m->length();
			local.forwardExpansions++;
			int8_t maxVal = 0;
			int8_t minVal = 0;
			
			for (CCMove *t = m; t; t = t->next)
			{
				cc.ApplyMove(s, t);
				uint64_t succ = cc.rank(s);
				cc.UndoMove(s, t);
				local.apply++;
				local.undo++;
				local.rank++;
				
				if (proverToMove)
				{
					minVal = std::min(minVal, result[succ]);
					if ((!strictDepth && result[succ] > 0) || // immediate win
						(strictDepth && (result[succ] == round+1)))
					{
						result[val] = round+2;//result[succ]+1;//
						local.winningStates++;
						local.changed++;
						proven = false;
						break;
					}
					if (result[succ] >= 0) // it might be round+2
						proven = false;
				}
				else {
					maxVal = std::max(maxVal, result[succ]);
					if ((!strictDepth && result[succ] < 0) ||
						(strictDepth && result[succ] == -(round+1)))
					{
						if (!winOnly)
						{
							result[val] = -(round+2);
							local.losingStates++;
							local.changed++;
						}
						proven = false;
						break;
					}
					else if (result[succ] <= 0)  // it might be -(round+2)
						proven = false;
				}
			}
			if ((!strictDepth && proverToMove && proven && (minVal < 0)) ||
				(strictDepth && proverToMove && proven && (minVal == -(round+1))))
			{
				if (!winOnly)
				{
					result[val] = -(round+2);
					local.losingStates++;
					local.changed++;
					
					// optionally back-prop to all parents
					if (backProp)
					{
						CCMove *back = cc.getReverseMoves(s);
						local.backwardExpansions++;
						local.backwardChildren += back->length();
						for (CCMove *nextBack = back; nextBack; nextBack = nextBack->next)
						{
							cc.UndoMove(s, nextBack);
							uint64_t parent = cc.rank(s);
							cc.ApplyMove(s, nextBack);
							local.undo++;
							local.apply++;
							local.rank++;
							
							if (result[parent] == 0)
							{
								result[parent] = -(round+3);
								local.losingStates++;
								local.changed++;
							}
						}
						cc.freeMove(back);
					}
				}
			}
			if ((!strictDepth && !proverToMove && proven && (maxVal > 0)) ||
				(strictDepth && !proverToMove && proven && (maxVal == (round+1))))
			{
				result[val] = (round+2);
				local.winningStates++;
				local.changed++;

				// optionally back-prop to all parents
				if (backProp)
				{
					CCMove *back = cc.getReverseMoves(s);
					local.backwardExpansions++;
					local.backwardChildren += back->length();
					for (CCMove *nextBack = back; nextBack; nextBack = nextBack->next)
					{
						cc.UndoMove(s, nextBack);
						uint64_t parent = cc.rank(s);
						cc.ApplyMove(s, nextBack);
						local.undo++;
						local.apply++;
						local.rank++;

						if (result[parent] == 0)
						{
							result[parent] = (round+3);
							local.winningStates++;
							local.changed++;

						}
					}
					cc.freeMove(back);
				}
				
			}
			cc.freeMove(m);
		}
	}
	
	stat += local;
}

template <bool strictDepth, bool winOnly, bool backProp>
void ForwardSolver<strictDepth, winOnly, backProp>::SolveGame(std::vector<int8_t> &wins, bool threaded)
{
	printf("--== Beginning forward solve ==--\n");
	printf("Strict Depth: %s; win only: %s; back prop: %s\n", strictDepth?"true":"false", winOnly?"true":"false", backProp?"true":"false");
	int threadCnt = std::thread::hardware_concurrency();
	if (!threaded)
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
	printf("Finding won positions for player %d\n", provingPlayer);
	float perc = 0;
	
	std::vector<std::thread *> threads(threadCnt);
	
	Timer t;
	t.StartTimer();
	for (int x = 0; x < threads.size(); x++)
	{
		threads[x] = new std::thread(&ForwardSolver<strictDepth, winOnly, backProp>::SetProvenStates, this, std::ref(wins), maxRank, x, threads.size(),
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
		
	stat.changed = 0;
	int round = 0;
	do {
		stat.changed = 0;
		t.StartTimer();
		// propagating wins
		for (int x = 0; x < threads.size(); x++)
		{
			threads[x] = new std::thread(&ForwardSolver<strictDepth, winOnly, backProp>::ProveStates, this, std::ref(wins), maxRank, x, threads.size(),
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
	
	std::cout << "strictDepth: " << strictDepth << " WinOnly " << winOnly << " backProp " << backProp << "\n";
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
