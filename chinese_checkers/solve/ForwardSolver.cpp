////
////  SimpleSolve.cpp
////  Solve Chinese Checkers
////
////  Created by Nathan Sturtevant on 7/23/11.
////  Copyright 2011 University of Denver. All rights reserved.
////
//
//#include "ForwardSolver.h"
//#include <stdio.h>
//#include <stdlib.h>
//#include <vector>
//#include <thread>
//#include <mutex>
//#include "Timer.h"
//#include "CCheckers.h"
//#include "SolveStats.h"
//
//void SetProvenStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
//					 int numThreads, int provingPlayer, stats &stat)
//{
//	CCheckers cc;
//	CCState s;
//	stats local;
//	for (uint64_t val = threadID; val < maxRank; val+=numThreads)
//	{
//		result[val] = 0;
//		if (cc.unrank(val, s))
//		{
//			local.legalStates++;
//			int winner = cc.Winner(s);
//			if (winner == provingPlayer)
//			{
//				result[val] = 1;
//				local.winningStates++;
//			}
//			else if (winner == 1-provingPlayer)
//			{
//				result[val] = -1;
//				local.losingStates++;
//			}
//		}
//	}
//	stat += local;
//}
//
//void ProveStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
//					 int numThreads, int provingPlayer, stats &stat, int round)
//{
//	CCheckers cc;
//	CCState s;
//	stats local;
//	
//	for (uint64_t val = threadID; val < maxRank; val+=numThreads)
//	{
//		if (result[val] != 0) // already proven
//			continue;
//
//		local.unrank++;
//		if (cc.unrank(val, s))
//		{
//			bool proven = true;
//			bool proverToMove = (s.toMove==provingPlayer);
//
//			CCMove *m = cc.getMoves(s);
//			local.forwardChildren += m->length();
//			local.forwardExpansions++;
//			int8_t maxVal = 0;
//			int8_t minVal = 0;
//			
//			for (CCMove *t = m; t; t = t->next)
//			{
//				cc.ApplyMove(s, t);
//				uint64_t succ = cc.rank(s);
//				cc.UndoMove(s, t);
//				local.apply++;
//				local.undo++;
//				local.rank++;
//				
//				if (proverToMove)
//				{
//					minVal = std::min(minVal, result[succ]);
//#ifdef WINLOSS
//					if (result[succ] > 0) // immediate win
//#else
//					if (result[succ] == round+1) // immediate win
//#endif
//					{
//						result[val] = round+2;//result[succ]+1;//
//						local.winningStates++;
//						local.changed++;
//						proven = false;
//						break;
//					}
//					if (result[succ] >= 0) // it might be round+2
//						proven = false;
//				}
//				else {
//					maxVal = std::max(maxVal, result[succ]);
//#ifdef WINLOSS
//					if (result[succ] < 0) // immediate loss
//#else
//					if (result[succ] == -(round+1)) // immediate loss
//#endif
//					{
//						result[val] = -(round+2);
//						local.losingStates++;
//						local.changed++;
//						proven = false;
//						break;
//					}
//					else if (result[succ] <= 0)  // it might be -(round+2)
//						proven = false;
//				}
//			}
//#ifdef WINLOSS
//			if (proverToMove && proven && (minVal < 0))
//#else
//			if (proverToMove && proven && (minVal == -(round+1)))
//#endif
//				{
//				result[val] = -(round+2);
//					local.losingStates++;
//				local.changed++;
//			}
//#ifdef WINLOSS
//			if (!proverToMove && proven && (maxVal > 0))
//#else
//			if (!proverToMove && proven && (maxVal == (round+1)))
//#endif
//			{
//				result[val] = (round+2);
//				local.winningStates++;
//				local.changed++;
//			}
//			cc.freeMove(m);
//		}
//	}
//	
//	stat += local;
//}
//
//void ForwardSolveGame(std::vector<int8_t> &wins)
//{
//	printf("--== Beginning forward solve ==--\n");
//	int provingPlayer = 0;
//	CCheckers cc;
//	CCState s;
//	Timer total;
//	total.StartTimer();
//
//	cc.Reset(s);
//	uint64_t root = cc.rank(s);
//	uint64_t maxRank = cc.getMaxRank();
//	wins.clear();
//	wins.resize(maxRank);
//
//	stats stat;
//	printf("Finding won positions for player %d\n", provingPlayer);
//	float perc = 0;
//	
//	std::vector<std::thread *> threads(std::thread::hardware_concurrency());
//	
//	Timer t;
//	t.StartTimer();
//	for (int x = 0; x < threads.size(); x++)
//	{
//		threads[x] = new std::thread(SetProvenStates, std::ref(wins), maxRank, x, threads.size(),
//									 provingPlayer, std::ref(stat));
//	}
//	for (int x = 0; x < threads.size(); x++)
//	{
//		threads[x]->join();
//		delete threads[x];
//		threads[x] = 0;
//	}
//	t.EndTimer();
//	printf("%1.3fs elapsed\n", t.GetElapsedTime());
//	std::cout << maxRank << " of " << maxRank << " 100% complete " <<
//	stat.legalStates << " legal " << stat.winningStates << " winning " << stat.illegalStates << " illegal" << std::endl;
//
//	stat.changed = 0;
//	int round = 0;
//	do {
//		stat.changed = 0;
//		t.StartTimer();
//		// propagating wins
//		for (int x = 0; x < threads.size(); x++)
//		{
//			threads[x] = new std::thread(ProveStates, std::ref(wins), maxRank, x, threads.size(),
//										 provingPlayer, std::ref(stat), round);
//		}
//		for (int x = 0; x < threads.size(); x++)
//		{
//			threads[x]->join();
//			delete threads[x];
//			threads[x] = 0;
//		}
//		t.EndTimer();
//		
//		printf("round %d; %llu changed; expansions: %llu (f) %llu (b); %llu of %llu proven (%llu wins %llu losses); %1.3fs elapsed\n", round++,
//			   stat.changed, stat.forwardExpansions, stat.backwardExpansions, stat.winningStates+stat.losingStates, maxRank, stat.winningStates, stat.losingStates, t.GetElapsedTime());
//	} while (stat.changed != 0);
//
//	total.EndTimer();
//	printf("%1.3fs elapsed\n", total.GetElapsedTime());
//	
//	if (wins[root])
//	{
//		printf("Win proven for player %d\n", provingPlayer);
//	}
//}
//
