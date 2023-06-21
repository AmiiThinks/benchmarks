//
//  HybridBitSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 8/21/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef HybridBitSolver_h
#define HybridBitSolver_h

#include <stdio.h>
#include "NBitArray.h"
#include <thread>
#include <mutex>
#include <cassert>
#include "SharedQueue.h"
#include "CCheckers.h"
#include "Timer.h"
#include "SolveStats.h"

namespace HybridBitSolver {
	
	const int provingPlayer = 0;
	const int otherPlayer = 1;
	
	template <int coarseListSize = 1024>
	class HybridBitSolver
	{
	public:
		void SolveGame(NBitArray<2> &res, bool threaded = true);
	private:
		void InitializeData(NBitArray<2> &results);
		void ProveData(NBitArray<2> &results);
		
		void GetParents(CCheckers &cc, CCState &s, std::vector<uint64_t> &parents);
		void FlushData(int threshold, std::vector<uint64_t> &wins, std::vector<uint64_t> &losses, std::vector<uint64_t> &parents, NBitArray<2> &results, stats &local);
		
		std::mutex resultLock;
		std::mutex coarseLock;
		enum coarseStatus {
			kHasProvableStates = 0x1,
			kHasProvenStates = 0x2
		};
		NBitArray<2> coarseData;
		SharedQueue<std::pair<uint64_t, uint64_t>> queue;
		stats stat;
	};
	
	enum gameResult {
		kUnknown = 0,
		kWin = 1,
		kLoss = 2,
		kDraw = 3
	};
	
	template <int coarseListSize>
	void HybridBitSolver<coarseListSize>::SolveGame(NBitArray<2> &res, bool threaded)
	{
		int threadCnt = std::thread::hardware_concurrency();
		if (!threaded)
			threadCnt = 1;
		printf("Running with %d threads\n", threadCnt);
		
		
		CCheckers cc;
		CCState s;
		Timer total, t;
		total.StartTimer();
		
		cc.Reset(s);
		uint64_t maxRank = cc.getMaxRank();
		uint64_t coarseEntries = maxRank/coarseListSize+1; // might take an extra 2 bits! ;)
		
		res.Resize(maxRank);
		res.Clear();
		
		coarseData.Resize(coarseEntries);
		coarseData.Clear();
		
		t.StartTimer();
		std::vector<std::thread> threads(threadCnt);
		for (int x = 0; x < threads.size(); x++)
		{
			threads[x] = std::thread(&HybridBitSolver<coarseListSize>::InitializeData, this, std::ref(res));
		}
		for (uint64_t x = 0; x < maxRank; x += coarseListSize)
			queue.WaitAdd({x, std::min(x+coarseListSize, maxRank)});
		for (int x = 0; x < threads.size(); x++)
			queue.WaitAdd({0, 0});
		for (int x = 0; x < threads.size(); x++)
		{
			threads[x].join();
		}
		t.EndTimer();
		
		printf("Init; %llu changed; expansions: %llu (f) %llu (b); %llu of %llu proven (%llu wins %llu losses); %1.3fs elapsed\n",
			   stat.changed, stat.forwardExpansions, stat.backwardExpansions, stat.winningStates+stat.losingStates, maxRank, stat.winningStates, stat.losingStates, t.GetElapsedTime());
		
		
		/****************************************************************************************************************************************************************************
		 ****************************************************************************************************************************************************************************
		 ****************************************************************************************************************************************************************************
		 * 1. Check whether threads are idle when plugged in to power                                                                                                               *
		 * 2. We are always only looking for wins at min nodes or lesses at max nodes, so we can stop analyzing a node when a single child doesn't have a value                     *
		 * 3. Check why so few states are proven (what happens if we lower the time for changing the input array) [May not matter in the end - more layers but much less expensive] *
		 ****************************************************************************************************************************************************************************
		 ****************************************************************************************************************************************************************************
		 ****************************************************************************************************************************************************************************
		 */
		
		stat.changed = 0;
		int round = 0;
		do {
			stat.changed = 0;
			t.StartTimer();
			// propagating wins
			for (int x = 0; x < threads.size(); x++)
			{
				threads[x] = std::thread(&HybridBitSolver<coarseListSize>::ProveData, this, std::ref(res));
			}
			
			uint64_t lowStart = 0;
			uint64_t highStart = maxRank-(maxRank%coarseListSize);
			while (lowStart <= highStart)
			{
				if (coarseData.Get(lowStart/coarseListSize)&kHasProvableStates)
					queue.WaitAdd({lowStart, std::min(lowStart+coarseListSize, maxRank)});
				if (coarseData.Get(highStart/coarseListSize)&kHasProvableStates)
					queue.WaitAdd({highStart, std::min(highStart+coarseListSize, maxRank)});
				lowStart += coarseListSize;
				highStart -= coarseListSize;
			}
			//		for (uint64_t x = 0; x < maxRank; x += coarseListSize)
			//			if (coarseData.Get(x/coarseListSize)&kHasProvableStates)
			//				queue.WaitAdd({x, std::min(x+coarseListSize, maxRank)});
			for (int x = 0; x < threads.size(); x++)
				queue.WaitAdd({0, 0});
			for (int x = 0; x < threads.size(); x++)
			{
				threads[x].join();
			}
			t.EndTimer();
			
			printf("Round %d; %llu changed; expansions: %llu (f) %llu (b); %llu of %llu proven (%llu wins %llu losses); %1.3fs elapsed\n", round++,
				   stat.changed, stat.forwardExpansions, stat.backwardExpansions, stat.winningStates+stat.losingStates, maxRank, stat.winningStates, stat.losingStates, t.GetElapsedTime());
		} while (stat.changed != 0);
		
		std::cout << stat << "\n";
		
		total.EndTimer();
		printf("%1.2fs elapsed\n", total.EndTimer());
	}
	
	template <int coarseListSize>
	void HybridBitSolver<coarseListSize>::GetParents(CCheckers &cc, CCState &s, std::vector<uint64_t> &parents)
	{
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *tmp = m; tmp != 0; tmp = tmp->next)
		{
			cc.UndoMove(s, tmp);
			parents.push_back(cc.rank(s));
			cc.ApplyMove(s, tmp);
		}
		cc.freeMove(m);
	}
	
	template <int coarseListSize>
	void HybridBitSolver<coarseListSize>::FlushData(int threshold, std::vector<uint64_t> &wins, std::vector<uint64_t> &losses, std::vector<uint64_t> &parents, NBitArray<2> &results, stats &local)
	{
		if (parents.size() > threshold)
		{
			coarseLock.lock();
			for (auto i : parents)
			{
				coarseData.Or(i/coarseListSize, kHasProvableStates);
			}
			coarseLock.unlock();
			parents.resize(0);
		}
		
		if (wins.size()+losses.size() > threshold)
		{
			resultLock.lock();
			for (auto i : wins)
			{
				if (results.Get(i) == kUnknown)
				{
					local.winningStates++;
					local.changed++;
					results.Set(i, kWin);
				}
			}
			for (auto i : losses)
			{
				if (results.Get(i) == kUnknown)
				{
					local.losingStates++;
					local.changed++;
					results.Set(i, kLoss);
				}
			}
			resultLock.unlock();
			
			wins.resize(0);
			losses.resize(0);
		}
		
	}
	
	
	template <int coarseListSize>
	void HybridBitSolver<coarseListSize>::InitializeData(NBitArray<2> &results)
	{
		stats local;
		CCheckers cc;
		CCState s;
		std::pair<uint64_t, uint64_t> work;
		std::vector<uint64_t> parentsOfKnownStates, tmp;
		std::vector<uint64_t> wins, losses;
		
		while (true) {
			queue.WaitRemove(work);
			if (work.first == work.second)
				break;
			
			for (uint64_t val = work.first; val < work.second; val++)
			{
				if (cc.unrank(val, s))
				{
					local.legalStates++;
					int winner = cc.Winner(s);
					if (winner == provingPlayer)
					{
						wins.push_back(val);
						GetParents(cc, s, tmp);
						wins.insert(wins.end(), tmp.begin(), tmp.end());
						parentsOfKnownStates.insert(parentsOfKnownStates.end(), tmp.begin(), tmp.end());
						tmp.clear();
					}
					else if (winner == otherPlayer)
					{
						losses.push_back(val);
						GetParents(cc, s, tmp);
						losses.insert(losses.end(), tmp.begin(), tmp.end());
						parentsOfKnownStates.insert(parentsOfKnownStates.end(), tmp.begin(), tmp.end());
						tmp.clear();
					}
				}
				else {
					assert(!"Unranking failed");
				}
			}
			FlushData(1024, wins, losses, parentsOfKnownStates, results, local);
		}
		FlushData(0, wins, losses, parentsOfKnownStates, results, local);
		
		stat += local;
	}
	
	template <int coarseListSize>
	void HybridBitSolver<coarseListSize>::ProveData(NBitArray<2> &results)
	{
		stats local;
		CCheckers cc;
		CCState s;
		std::pair<uint64_t, uint64_t> work;
		std::vector<uint64_t> parentsOfKnownStates;
		std::vector<uint64_t> wins, losses;
		std::vector<uint64_t> tmpParents;
		
		while (true) {
			queue.WaitRemove(work);
			if (work.first == work.second)
				break;
			
			for (uint64_t val = work.first; val < work.second; val++)
			{
				if (results.Get(val) != kUnknown)
					continue;
				if (cc.unrank(val, s))
				{
					int totalSucc = 0, lossSucc = 0, wonSucc = 0;
					CCMove *m = cc.getMoves(s);
					for (CCMove *t = m; t; t = t->next)
					{
						cc.ApplyMove(s, t);
						uint64_t rank = cc.rank(s);
						cc.UndoMove(s, t);
						totalSucc++;
						switch (results.Get(rank))
						{
							case kWin: wonSucc++; break;
							case kLoss: lossSucc++; break;
						}
						//					if ((s.toMove == provingPlayer && wonSucc > 0) || (s.toMove == otherPlayer && lossSucc > 0))
						//						printf("Immediate proven win/loss (should never get here)\n");
						if (s.toMove == provingPlayer && lossSucc < totalSucc)
							break;
						if (s.toMove == otherPlayer && wonSucc < totalSucc)
							break;
					}
					cc.freeMove(m);
					
					
					if (s.toMove == otherPlayer && wonSucc == totalSucc)
					{
						wins.push_back(val);
						
						// won min node - parents get immediate wins
						GetParents(cc, s, tmpParents);
						wins.insert(wins.end(), tmpParents.begin(), tmpParents.end());
						parentsOfKnownStates.insert(parentsOfKnownStates.end(), tmpParents.begin(), tmpParents.end());
						tmpParents.clear();
					}
					else if (s.toMove == provingPlayer && wonSucc > 0)
					{
						wins.push_back(val);
						GetParents(cc, s, parentsOfKnownStates);
					}
					else if (s.toMove == otherPlayer && lossSucc > 0)
					{
						losses.push_back(val);
						GetParents(cc, s, parentsOfKnownStates);
					}
					else if (s.toMove == provingPlayer && lossSucc == totalSucc)
					{
						losses.push_back(val);
						
						// lost max node - parents get immediate losses
						GetParents(cc, s, tmpParents);
						losses.insert(losses.end(), tmpParents.begin(), tmpParents.end());
						parentsOfKnownStates.insert(parentsOfKnownStates.end(), tmpParents.begin(), tmpParents.end());
						tmpParents.clear();
					}
				}
				else {
					assert(!"Unranking failed");
				}
			}
			FlushData(32, wins, losses, parentsOfKnownStates, results, local);
		}
		
		FlushData(0, wins, losses, parentsOfKnownStates, results, local);
		stat += local;
	}
	
}
	
#endif /* HybridBitSolver_h */
