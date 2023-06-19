////
////  BackwardSolver.cpp
////  CC Solver
////
////  Created by Nathan Sturtevant on 4/22/16.
////  Copyright © 2016 NS Software. All rights reserved.
////
//
//#include <stdio.h>
//#include "BackwardSolver.h"
//#include <stdio.h>
//#include <stdlib.h>
//#include <vector>
//#include "CCheckers.h"
//#include <thread>
//#include <mutex>
//#include "Timer.h"
//
//// TODO:
//// 1) Mark states with fully solved parents
//// 2) Find way to avoid unranking all solved states
//// 3) Only check changed states
//
//namespace BackwardSolver {
//	
//	
//	void SetProvenStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
//						 int numThreads, int provingPlayer, stats &stat)
//	{
//		CCheckers cc;
//		CCState s;
//		uint64_t legalStates = 0;
//		uint64_t winningStates = 0;
//		uint64_t losingStates = 0;
//		uint64_t illegalStates = 0;
//		for (uint64_t val = threadID; val < maxRank; val+=numThreads)
//		{
//			result[val] = 0;
//			if (cc.unrank(val, s))
//			{
//				legalStates++;
//				if (cc.Done(s))
//				{
//					if (cc.Winner(s) == provingPlayer)
//					{
//						if (s.toMove == 1-provingPlayer)
//						{
//							result[val] = 1;
//							winningStates++;
//						}
//						else {
//							illegalStates++;
//						}
//					}
//					if (cc.Winner(s) == 1-provingPlayer)
//					{
//						if (s.toMove == provingPlayer)
//						{
//							result[val] = -1;
//							losingStates++;
//						}
//						else {
//							illegalStates++;
//						}
//					}
//
//				}
//			}
//			else {
//			}
//		}
//		stat.lock.lock();
//		stat.legalStates += legalStates;
//		stat.winningStates += winningStates;
//		stat.losingStates += winningStates;
//		stat.illegalStates += illegalStates;
//		stat.lock.unlock();
//	}
//	
//	void ProveStates(std::vector<int8_t> &result, uint64_t maxRank, int threadID,
//					 int numThreads, int provingPlayer, stats &stat, int round)
//	{
//		CCheckers cc;
//		CCState s;
//		uint64_t legalStates = 0;
//		uint64_t winningStates = 0;
//		uint64_t losingStates = 0;
//		uint64_t illegalStates = 0;
//		uint64_t forwardExpansions = 0;
//		uint64_t backwardExpansions = 0;
//		uint64_t forwardChildren = 0;
//		uint64_t backwardChildren = 0;
//		uint64_t changed = 0;
//		
//		for (uint64_t val = 0; val < maxRank; val++)
//		{
//#ifdef WINLOSS
//			if (result[val] >= round+1) // proven win
//#else
//			if (result[val] == round+1) // proven win
//#endif
//			{
//				cc.unrank(val, s);
//
//				CCMove *backMoves = cc.getReverseMoves(s);
//				backwardChildren += backMoves->length();
//				backwardExpansions++;
//				for (CCMove *t = backMoves; t; t = t->next)
//				{
//					cc.UndoMove(s, t);
//					
//					uint64_t parentRank = cc.rank(s);
//					if ((parentRank%numThreads) == threadID && result[parentRank] == 0 && cc.Winner(s) != 1-provingPlayer)
//					{
//						bool done = false;
//						bool proven = true;
//						bool proverToMove = (s.toMove==provingPlayer);
//						
//						if (proverToMove) // immediate proof max nodes from winning child
//						{
//							result[parentRank] = round+2;
//							changed++;
//							winningStates++;
//						}
//						else {
//							
//							CCMove *forwardMoves = cc.getMoves(s);
//							forwardChildren += forwardMoves->length();
//							forwardExpansions++;
//							
//							for (CCMove *next = forwardMoves; next && !done; next = next->next)
//							{
//								cc.ApplyMove(s, next);
//								
//								uint64_t succ = cc.rank(s);
//								if ((proverToMove) && result[succ] > 0)
//								{
//									proven = true;
//									done = true;
//								}
//								if ((!proverToMove) && (result[succ] <= 0))
//								{
//									done = true;
//									proven = false;
//								}
//								
//								cc.UndoMove(s, next);
//							}
//							if (done == false || proven == true)
//							{
//								result[parentRank] = round+2;
//								winningStates++;
//								changed++;
//							}
//							
//							cc.freeMove(forwardMoves);
//						}
//					}
//					
//					
//					cc.ApplyMove(s, t);
//				}
//				cc.freeMove(backMoves);
//			}
//			
//#ifdef WINLOSS
//			if (result[val] <= -(round+1)) // proven loss
//#else
//			if (result[val] == -(round+1)) // proven loss
//#endif
//			{
//				cc.unrank(val, s);
//				
//				CCMove *backMoves = cc.getReverseMoves(s);
//				backwardChildren += backMoves->length();
//				backwardExpansions++;
//				
//				for (CCMove *t = backMoves; t; t = t->next)
//				{
//					cc.UndoMove(s, t);
//					
//					uint64_t parentRank = cc.rank(s);
//					if ((parentRank%numThreads) == threadID && result[parentRank] == 0 && cc.Winner(s) != provingPlayer)
//					{
//						bool done = false;
//						bool proven = true;
//						bool proverToMove = (s.toMove==provingPlayer);
//						
//						if (!proverToMove && (parentRank%numThreads) == threadID) // immediate proof min nodes from losing child
//						{
//							result[parentRank] = -(round+2);
//							changed++;
//							losingStates++;
//						}
//						else {
//							
//							CCMove *forwardMoves = cc.getMoves(s);
//							forwardChildren += forwardMoves->length();
//							forwardExpansions++;
//							
//							for (CCMove *next = forwardMoves; next && !done; next = next->next)
//							{
//								cc.ApplyMove(s, next);
//								
//								uint64_t succ = cc.rank(s);
//								if ((!proverToMove) && result[succ] < 0)
//								{
//									proven = true;
//									done = true;
//								}
//								if ((proverToMove) && (result[succ] >= 0))
//								{
//									done = true;
//									proven = false;
//								}
//								
//								cc.UndoMove(s, next);
//							}
//							if (done == false || proven == true)
//							{
//								result[parentRank] = -(round+2);
//								losingStates++;
//								changed++;
//							}
//							
//							cc.freeMove(forwardMoves);
//						}
//					}
//					
//					
//					cc.ApplyMove(s, t);
//				}
//				cc.freeMove(backMoves);
//			}
//		}
//		
//		stat.lock.lock();
//		stat.legalStates += legalStates;
//		stat.winningStates += winningStates;
//		stat.losingStates += losingStates;
//		stat.illegalStates += illegalStates;
//		stat.forwardExpansions += forwardExpansions;
//		stat.backwardExpansions += backwardExpansions;
//		stat.changed += changed;
//		stat.lock.unlock();
//	}
//	
//}
//
//
//void BackwardSolveGame(std::vector<int8_t> &wins)
//{
//	printf("--== Beginning backward solve ==--\n");
//	printf("Running with %d threads\n", std::thread::hardware_concurrency());
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
//	BackwardSolver::stats stat;
//
//	printf("Finding all terminal positions\n");
//	
//	std::vector<std::thread *> threads(std::thread::hardware_concurrency());
//	
//	Timer t;
//	t.StartTimer();
//	for (int x = 0; x < threads.size(); x++)
//	{
//		threads[x] = new std::thread(BackwardSolver::SetProvenStates, std::ref(wins), maxRank, x, threads.size(),
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
//	stat.legalStates << " legal " << stat.winningStates << " winning " << stat.losingStates << " losing " << stat.illegalStates << " illegal" << std::endl;
//	
//	//	printf("%lld states unranked; %lld were winning; %lld were tech. illegal\n",
//	//		   stat.legalStates, stat.winningStates, stat.illegalStates);
//	
//	stat.changed = 0;
//	int round = 0;
//	do {
//		stat.changed = 0;
//		t.StartTimer();
//		// propagating wins
//		for (int x = 0; x < threads.size(); x++)
//		{
//			threads[x] = new std::thread(BackwardSolver::ProveStates, std::ref(wins), maxRank, x, threads.size(),
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
//	
//	uint64_t w=0,l=0,d=0;
//	for (uint64_t x = 0; x < wins.size(); x++)
//	{
//		if (wins[x] > 0)
//			w++;
//		else if (wins[x] < 0)
//			l++;
//		else if (wins[x] == 0)
//		{
//			d++;
//			cc.unrank(x, s);
//			s.Print();
//			printf("State %llu unproven\n", x);
//		}
//	}
//	printf("%llu wins, %llu losses; %llu draws\n", w, l, d);
//}
//
