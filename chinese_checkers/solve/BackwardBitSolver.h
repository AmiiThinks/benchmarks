//
//  BackwardBitSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 4/22/16.
//  Copyright © 2016 NS Software. All rights reserved.
//

#ifndef BackwardBitSolver_h
#define BackwardBitSolver_h

#include <stdint.h>
#include <vector>
#include <thread>
#include "SolveStats.h"
#include "Timer.h"
#include "CCheckers.h"
#include "NBitArray.h"

namespace BackwardBitSolver {
	const int win = 2;
	const int loss = 0;
	const int draw = 1;
	
	const char *resStr(int v)
	{
		switch (v)
		{
			case win: return "win";
			case loss: return "loss";
			case draw: return "draw";
			default: return "unknown";
		}
	}
	
	const int backBit = 1;
	const int forwardBit = 2;
	
	std::mutex printLock;
	
	template <bool winOnly = false>
	class BackwardBitSolver
	{
	public:
		BackwardBitSolver(const char *path, bool concurrent = true);
		int Lookup(const CCState &s) const;
		int Lookup(uint64_t rank) const;
		void PrintStats() const;
	private:
		void SolveGame(NBitArray<2> &result, bool concurrent = true);
		void SetProvenStates(NBitArray<2> &result, uint64_t maxRank, int threadID,
							 int numThreads, int provingPlayer, stats &stat);
		
		void ExpandBackward(NBitArray<2> &result, uint64_t maxRank, int threadID,
							int numThreads, int provingPlayer, stats &stat, int round);
		void ExpandForward(NBitArray<2> &result, uint64_t maxRank, int threadID,
						   int numThreads, int provingPlayer, stats &stat, int round);
		
		const char *GetFileName(const char *path);

		NBitArray<2> flags;
		NBitArray<2> data;
		std::mutex flagLock;
	};
	
	template <bool winOnly>
	void BackwardBitSolver<winOnly>::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case win: w++; break;
				case loss: l++; break;
				case draw: d++; break;
			}
		}
		printf("--BBS Data Summary--\n");
		printf("%llu wins, %llu losses, %llu draws\n", w, l, d);
	}

	template <bool winOnly>
	BackwardBitSolver<winOnly>::BackwardBitSolver(const char *path, bool concurrent)
	{
		FILE *f = fopen(GetFileName(path), "r");
		if (f == 0)
		{
			SolveGame(data, concurrent);
			data.Write(GetFileName(path));
		}
		else {
			data.Read(f);
		}
	}

	template <bool winOnly>
	const char *BackwardBitSolver<winOnly>::GetFileName(const char *path)
	{
		static std::string s;
		s = path;
		s += "CC-SOLVE-BACKBIT-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += ".dat";
		return s.c_str();
	}

	template <bool winOnly>
	int BackwardBitSolver<winOnly>::Lookup(const CCState &s) const
	{
		
		CCheckers cc;
		uint64_t v = cc.rank(s);
		return data.Get(v);
	}
	
	template <bool winOnly>
	int BackwardBitSolver<winOnly>::Lookup(uint64_t rank) const
	{
		return data.Get(rank);
	}

	template <bool winOnly>
	void BackwardBitSolver<winOnly>::SetProvenStates(NBitArray<2> &result, uint64_t maxRank, int threadID,
													 int numThreads, int provingPlayer, stats &stat)
	{
		CCheckers cc;
		CCState s;
		stats local;
		std::vector<std::pair<int64_t, int>> toWrite;
		for (uint64_t val = threadID; val < maxRank; val+=numThreads)
		{
			if (toWrite.size() > 512)
			{
				flagLock.lock();
				for (auto &value : toWrite)
				{
					flags.Set(value.first, backBit);
					result.Set(value.first, value.second);
				}
				flagLock.unlock();
				toWrite.clear();
			}
			
			//result.Set(val, 0);
			//result[val] = 0;
			if (cc.unrank(val, s))
			{
				local.legalStates++;
				if (cc.Done(s))
				{
					if (cc.Winner(s) == provingPlayer)
					{
						if (s.toMove == 1-provingPlayer)
						{
							//result[val] = win;
							//result.Set(val, win);
							toWrite.push_back({val, win});
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
							result.Set(val, loss);
							//result[val] = loss;
							local.losingStates++;
							toWrite.push_back({val, loss});
						}
						else {
							local.illegalStates++;
						}
					}
					
				}
			}
		}
		if (toWrite.size() > 0)
		{
			flagLock.lock();
			for (auto &value : toWrite)
			{
				flags.Set(value.first, backBit);
				result.Set(value.first, value.second);
			}
			flagLock.unlock();
			toWrite.clear();
		}
		
		stat += local;
	}
	
	template <bool winOnly>
	void BackwardBitSolver<winOnly>::ExpandBackward(NBitArray<2> &result, uint64_t maxRank, int threadID,
													int numThreads, int provingPlayer, stats &stat, int round)
	{
		CCheckers cc;
		CCState s;
		stats local;
		std::vector<int64_t> forwardCache;
		std::vector<int64_t> backwardClear;
		
		for (uint64_t val = threadID; val < maxRank; val+=numThreads)
		{
			if (flags[val]&backBit)
			{
				//printf("--- %lld set\n", val);
				backwardClear.push_back(val);
				cc.unrank(val, s);
				local.unrank++;
				CCMove *back = cc.getReverseMoves(s);
				local.backwardExpansions++;
				local.backwardChildren += back->length();
				
				for (CCMove *t = back; t; t = t->next)
				{
					cc.UndoMove(s, t);
					int64_t parent = cc.rank(s);
					if (result[parent] == draw) // (drawn states are actually unknown) save adding proven states
						forwardCache.push_back(parent);
					cc.ApplyMove(s, t);
					
					local.undo++;
					local.apply++;
					local.rank++;
				}
				cc.freeMove(back);
			}
			
			if (forwardCache.size() > 512 || backwardClear.size() > 512)
			{
				flagLock.lock();
				for (auto &value : forwardCache)
					flags.Set(value, flags.Get(value)|forwardBit);
				for (auto &value : backwardClear)
					flags.Set(value, flags.Get(value)&forwardBit);
				flagLock.unlock();
				forwardCache.clear();
				backwardClear.clear();
			}
		}
		
		if (forwardCache.size() > 0 || backwardClear.size() > 0)
		{
			flagLock.lock();
			for (auto &value : forwardCache)
				flags.Set(value, flags.Get(value)|forwardBit);
			for (auto &value : backwardClear)
				flags.Set(value, flags.Get(value)&forwardBit);
			flagLock.unlock();
			forwardCache.clear();
			backwardClear.clear();
		}
		//	printLock.lock();
		//	std::cout << local << "\n";
		//	printLock.unlock();
		stat += local;
	}
	
	template <bool winOnly>
	void BackwardBitSolver<winOnly>::ExpandForward(NBitArray<2> &result, uint64_t maxRank, int threadID,
												   int numThreads, int provingPlayer, stats &stat, int round)
	{
		CCheckers cc;
		CCState s;
		stats local;
		std::vector<std::pair<int64_t, int>> toWrite;
		std::vector<int64_t> forwardToClear;
		
		for (uint64_t val = threadID; val < maxRank; val+=numThreads)
		{
			if ((flags[val]&forwardBit) && result[val] == draw) // forward check on parents
			{
				forwardToClear.push_back(val);
				cc.unrank(val, s);
				local.unrank++;
				CCMove *fore = cc.getMoves(s);
				local.forwardExpansions++;
				local.forwardChildren += fore->length();
				bool maxToMove = s.toMove == provingPlayer;
				
				int minResult = 3;
				int maxResult = -1;
				for (CCMove *t = fore; t; t = t->next)
				{
					cc.ApplyMove(s, t);
					int64_t childRank = cc.rank(s);
					cc.UndoMove(s, t);
					local.undo++;
					local.apply++;
					local.rank++;
					
					int next = result[childRank];
					minResult = std::min(minResult, next);
					maxResult = std::max(maxResult, next);
					if (maxToMove && next == win)
					{
						toWrite.push_back({val, win});
						//printf("Immediate win!\n");
						break;
					}
					if (!maxToMove && next == loss && !winOnly)
					{
						toWrite.push_back({val, loss});
						//printf("Immediate loss!\n");
						break;
					}
				}
				if (maxToMove && maxResult == loss && !winOnly)
				{
					//printf("All children losses at max node\n");
					toWrite.push_back({val, loss});
				}
				else if (!maxToMove && minResult == win)
				{
					//printf("All children wins at min node\n");
					toWrite.push_back({val, win});
				}
				
				cc.freeMove(fore);
			}
			
			if (toWrite.size() > 512 || forwardToClear.size() > 512)
			{
				flagLock.lock();
				for (auto &value : toWrite)
				{
					flags.Set(value.first, backBit);
					result.Set(value.first, value.second);
					if (value.second == win)
						local.winningStates++;
					else if (value.second == loss)
						local.losingStates++;
					local.changed++;
				}
				for (auto &value : forwardToClear)
				{
					flags.Set(value, flags.Get(value)&backBit);
				}
				flagLock.unlock();
				toWrite.clear();
				forwardToClear.clear();
			}
		}
		
		if (toWrite.size() > 0 || forwardToClear.size() > 0)
		{
			flagLock.lock();
			for (auto &value : toWrite)
			{
				flags.Set(value.first, backBit);
				result.Set(value.first, value.second);
				if (value.second == win)
					local.winningStates++;
				else if (value.second == loss)
					local.losingStates++;
				local.changed++;
			}
			for (auto &value : forwardToClear)
			{
				flags.Set(value, flags.Get(value)&backBit);
			}
			flagLock.unlock();
			toWrite.clear();
			forwardToClear.clear();
		}
		
		//	printLock.lock();
		//	std::cout << local << "\n";
		//	printLock.unlock();
		stat += local;
	}
	
	template <bool winOnly>
	void BackwardBitSolver<winOnly>::SolveGame(NBitArray<2> &wins, bool concurrent)
	{
		printf("--== Beginning bitbackward solve ==--\n");
		printf("winsOnly: %s\n", winOnly?"true":"false");
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
		
		wins.Resize(maxRank);
		flags.Resize(maxRank);
		for (int64_t x = 0; x < wins.Size(); x++)
		{
			wins.Set(x, draw);
			flags.Set(x, 0);
		}
		
		stats stat;
		
		printf("Finding all terminal positions\n");
		
		std::vector<std::thread *> threads(threadCnt);
		
		Timer t;
		t.StartTimer();
		for (int x = 0; x < threads.size(); x++)
		{
			threads[x] = new std::thread(&BackwardBitSolver<winOnly>::SetProvenStates, this, std::ref(wins), maxRank, x, threads.size(),
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
		
		
		//	for (int64_t x = 0; x < wins.Size(); x++)
		//	{
		//		if (wins.Get(x) == win)
		//		{
		//			printf("Bit %lld set\n", x);
		//			if (flags.Get(x) != backBit)
		//				printf("!!!Value isn't set for backward reading!\n");
		//		}
		//	}
		
		stat.changed = 0;
		int round = 0;
		do {
			stat.changed = 0;
			t.StartTimer();
			
			//printf("Expanding backwards states\n");
			for (int x = 0; x < threads.size(); x++)
			{
				threads[x] = new std::thread(&BackwardBitSolver<winOnly>::ExpandBackward, this, std::ref(wins), maxRank, x, threads.size(),
											 provingPlayer, std::ref(stat), round);
			}
			for (int x = 0; x < threads.size(); x++)
			{
				threads[x]->join();
				delete threads[x];
				threads[x] = 0;
			}
			//		std::cout << stat << "\n";
			//printf("Expanding forward states\n");
			for (int x = 0; x < threads.size(); x++)
			{
				threads[x] = new std::thread(&BackwardBitSolver<winOnly>::ExpandForward, this, std::ref(wins), maxRank, x, threads.size(),
											 provingPlayer, std::ref(stat), round);
			}
			for (int x = 0; x < threads.size(); x++)
			{
				threads[x]->join();
				delete threads[x];
				threads[x] = 0;
			}
			
			t.EndTimer();
			
			//		std::cout << stat << "\n";
			printf("round %d; %llu changed; expansions: %llu (f) %llu (b); %llu of %llu proven (%llu wins %llu losses); %1.3fs elapsed\n", round++,
				   stat.changed, stat.forwardExpansions, stat.backwardExpansions, stat.winningStates+stat.losingStates, maxRank, stat.winningStates, stat.losingStates, t.GetElapsedTime());
		} while (stat.changed != 0);
		std::cout << stat << "\n";
		
		total.EndTimer();
		printf("%1.3fs elapsed\n", total.GetElapsedTime());
		
		if (wins[root] == win)
		{
			printf("Win proven for player %d\n", provingPlayer);
		}
		
		
		uint64_t w=0,l=0,d=0;
		for (int64_t x = 0; x < wins.Size(); x++)
		{
			if (wins[x] == win)
				w++;
			else if (wins[x] == loss)
				l++;
			else if (wins[x] == draw)
			{
				d++;
				//			cc.unrank(x, s);
				//			s.Print();
				//			printf("State %llu unproven\n", x);
			}
		}
		printf("%llu wins, %llu losses; %llu draws\n", w, l, d);
	}
	
}
#endif /* BackwardBitSolver_h */
