//
//  ParallelSolver.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "InitialParallelSolver.h"
#include <cassert>
#include <thread>
#include <mutex>
#include "Timer.h"

namespace ParallelOne {
	static const char *resultText[] = {"Draw", "Loss", "Win", "Illegal"};
	
	Solver::Solver(const char *path, bool forceBuild)
	:workQueue(WORK_QUEUE_BUFFER), doneQueue(128), startQueue(128) // At most 128 threads?
	{
		if (NUM_PIECES == 2 || NUM_PIECES == 5)
		{
			printf("%d pieces not symmetric; aborting\n", NUM_PIECES);
			exit(0);
		}
		if (forceBuild)
		{
			BuildData(path, std::thread::hardware_concurrency());
		}
		else {
			FILE *f = fopen(GetFileName(path), "r");
			if (f == 0)
			{
				BuildData(path, std::thread::hardware_concurrency());
			}
			else {
				InitMetaData();
				data.Read(f);
			}
		}
	}
	
	void Solver::InitMetaData()
	{
		CCheckers cc;
		CCState s;
		int64_t memoryOffset = 0;
		uint64_t maxRank = r.getMaxRank();
		groups = new GroupStructure[r.getMaxP1Rank()];
		open.resize(r.getMaxP1Rank());
		memoryMult = r.getMaxP2Rank();
		// Note that in single-player CC we have to (maybe?) generate
		// the children of both the regular and flipped states
		
		int64_t totalStates = 0;
		symmetricStates = 0;
		for (int32_t x = 0; x < r.getMaxP1Rank(); x++)
		{
			open[x] = false;
			r.unrank(x, 0, s);
			CCState sym;
			cc.FlipPlayer(s, sym, 0);
			
			// Since we are only going up to 7x7 (6 piece) CC, we can narrow to 32 bits here
			int32_t otherRank = static_cast<int32_t>(r.rankP1(sym));
			if (otherRank < x) // we are redundant
			{
				groups[x].symmetricRank = -1;
			}
			else { // we aren't redundant
				groups[x].symmetricRank = otherRank; // this is our redundant partner
				groups[x].memoryOffset = memoryOffset;
				memoryOffset++;
				symmetricStates += r.getMaxP2Rank();
			}
			totalStates += r.getMaxP2Rank();
		}
		data.Resize(symmetricStates);
		data.Clear();
		
		printf("--> SYMMETRY: %lld total; %lld after symmetry reduction\n", totalStates, symmetricStates);
		
		printf("Starting single-agent BFS\n");
		Timer t;
		t.StartTimer();
		DoBFS();
		t.EndTimer();
		printf("%1.2fs in BFS\n", t.GetElapsedTime());
		GetSearchOrder();
	}
	
	void Solver::GetSearchOrder()
	{
		int d = 0;
		while (order.size() != r.getMaxP1Rank())
		{
			for (int x = bfs.size()-1; x >= 0; x--)
			{
				if (bfs[x] == d)
					order.push_back(x);
			}
			d++;
		}
	}
	
	void Solver::DoBFS()
	{
		CCheckers cc;
		CCState s;
		bfs.resize(r.getMaxP1Rank());
		std::fill(bfs.begin(), bfs.end(), -1);
		bfs[r.getMaxP1Rank()-1] = 0;
		
		int depth = 0;
		int written = 0;
		int total = 1;
		do {
			written = 0;
			for (int x = 0; x < r.getMaxP1Rank(); x++)
			{
				if (bfs[x] == depth)
				{
					r.unrankP1(x, s);
					CCMove *m = cc.getMoves(s);
					//					printf("%d ", x); s.PrintASCII();
					for (CCMove *tmp = m; tmp; tmp = tmp->next)
					{
						cc.ApplyMove(s, tmp);
						s.toMove = kMaxPlayer;
						int64_t rank = r.rankP1(s);
						//						printf("%d ", rank); s.PrintASCII();
						if (bfs[rank] == -1)
						{
							bfs[rank] = depth+1;
							written++;
						}
						s.toMove = kMinPlayer;
						cc.UndoMove(s, tmp);
					}
					cc.freeMove(m);
				}
			}
			total += written;
			printf("Depth %d complete. %d new. %d of %lld complete\n", depth, written, total, r.getMaxP1Rank());
			depth++;
		} while (written != 0);
	}
	
	void Solver::Initial(int numThreads)
	{
		CCheckers cc;
		CCState s, tmp;
		uint64_t maxRank = r.getMaxRank();
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();
		std::vector<result> wins;
		
		//		for (uint64_t x = 0; x < maxRank; x++)
		for (int64_t p1Rank = 0; p1Rank < max1; p1Rank++)
		{
			if (groups[p1Rank].symmetricRank == -1)
				continue;
			
			r.unrankP1(p1Rank, s);
			
			int startCount = cc.GetNumPiecesInStart(s, 0);
			int goalCount = cc.GetNumPiecesInGoal(s, 0);
			// 1. if no pieces in home, then only one possible goal.
			if (startCount == 0 && goalCount == 0)
			{
				r.unrankP2(0, s);
				stat.unrank++;
				
				if (cc.Winner(s) == -1)
					continue;
				data.Set(groups[p1Rank].memoryOffset*memoryMult, kLoss);
				open[p1Rank] = true;
				//groups[p1Rank].changed = true;
				proven++;
				PropagateWinToParent(cc, s, wins); // This is a win at the parent!
				WriteResult(wins, kWin);
				continue;
			}
			
			// 2. if pieces in home, then try all goals (could do better)
			for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
			{
				if (p2Rank > 0)
				{
					// clear p2 pieces
					for (int x = 0; x < NUM_PIECES; x++)
					{
						s.board[s.pieces[1][x]] = 0;
					}
				}
				r.unrankP2(p2Rank, s);
				int64_t x = r.rank(p1Rank, p2Rank);
				stat.unrank++;
				if (!cc.Legal(s))
				{
					if (data.Get(groups[p1Rank].memoryOffset*memoryMult+p2Rank) == kDraw)
					{
						proven++;
						open[p1Rank] = true;
						//groups[p1Rank].changed = true;
					}
					// TODO: Collect illegals!
					data.Set(groups[p1Rank].memoryOffset*memoryMult+p2Rank, kIllegal);
					continue;
				}
				switch (cc.Winner(s))
				{
					case -1: // no winner
						break;
					case 0: // not possible, because it's always player 0's turn
						// Actually, this is possible in one situation - but we consider
						// it illegal to make a suicide move to lose the game
						assert(!"(0) This isn't possible");
						break;
					case 1:
						data.Set(groups[p1Rank].memoryOffset*memoryMult+p2Rank, kLoss);
						proven++;
						//groups[p1Rank].changed = true;
						open[p1Rank] = true;
						tmp = s;
						PropagateWinToParent(cc, tmp, wins); // This is a win at the parent!
						WriteResult(wins, kWin);
						break;
				}
			}
		}
	}
	
	/*
	 * Note that s will be modified in this function
	 */
	void Solver::PropagateWinToParent(CCheckers &cc, CCState &s, std::vector<result> &wins)
	{
		CCState tmp;
		// Always called from Loss at max node
		assert(s.toMove == kMaxPlayer);
		stat.backwardExpansions++;
		
		// Flip the whole board then generate all moves. More efficient than
		// Generating moves and then flipping each time.
		cc.SymmetryFlipVert(s);
		assert(s.toMove == kMinPlayer);
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			int64_t p1Rank, p2Rank;
			
			cc.ApplyReverseMove(s, t);
			
			p1Rank = r.rankP1(s);
			
			stat.rank++;
			if (groups[p1Rank].symmetricRank != -1)
			{
				p2Rank = r.rankP2(s);
				
				//if (data.Get(groups[p1Rank].memoryOffset*memoryMult+p2Rank) == kDraw)
				{
					wins.push_back({static_cast<int32_t>(p1Rank), static_cast<int32_t>(p2Rank)});
//					proven++;
//					data.Set(groups[p1Rank].memoryOffset*memoryMult+p2Rank, kWin);
//					groups[p1Rank].changed = true;
				}
			}
			
			// Only flip pieces
			cc.SymmetryFlipHoriz_PO(s, tmp);
			
			p1Rank = r.rankP1(tmp);
			
			stat.rank++;
			if (groups[p1Rank].symmetricRank != -1)
			{
				p2Rank = r.rankP2(tmp);
				
				//if (data.Get(groups[p1Rank].memoryOffset*memoryMult+p2Rank) == kDraw)
				{
					wins.push_back({static_cast<int32_t>(p1Rank), static_cast<int32_t>(p2Rank)});
//					proven++;
//					data.Set(groups[p1Rank].memoryOffset*memoryMult+p2Rank, kWin);
//					groups[p1Rank].changed = true;
				}
			}
			cc.UndoReverseMove(s, t);
		}
		cc.freeMove(m);
	}
	
	
	/*
	 * Returns the value for the parent, so it has to be flipped.
	 */
	tResult Solver::GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip)
	{
		CCState tmp;
		// Max player to move
		if (doubleFlip)
			cc.SymmetryFlipHorizVert(s, tmp); // Same flip used initially to flip players - returns to initial p1rank
		else
			cc.SymmetryFlipVert(s, tmp); // Same flip used initially to flip players - returns to initial p1rank
		assert(tmp.toMove == kMinPlayer);
		
		CCMove *m = cc.getMoves(tmp);
		stat.forwardExpansions++;
		
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(tmp, t); // Now min player to move
			int64_t p1Rank, p2Rank;
			tResult val;
			// We re-use the initial p1Rank and just compute p2 here
			assert(tmp.toMove == kMaxPlayer);
			p1Rank = finalp1Rank;
			p2Rank = r.rankP2(tmp);
			val = Translate(kMaxPlayer, (tResult)data.Get(groups[p1Rank].memoryOffset*memoryMult+p2Rank)); // Invert back to get value for min player
			stat.rank++;
			cc.UndoMove(tmp, t);
			
			
			switch (val)
			{
				case kWin:
					if (tmp.toMove == kMaxPlayer)
					{
						assert(!"(a) Shouldn't get here.\n");
						cc.freeMove(m);
						return kWin;
					}
					break;
				case kLoss:
					if (tmp.toMove == kMinPlayer)
					{
						cc.freeMove(m);
						assert(!"(b) Shouldn't get here.\n");
						return kLoss;
					}
					break;
				case kDraw: // still unknown
					cc.freeMove(m);
					return kDraw;
					break;
				case kIllegal: // ignore
					break;
			}
			
		}
		cc.freeMove(m);
		return kLoss;
	}
	
	void Solver::SinglePassInnerLoop(CCheckers &cc, int64_t p1Rank, int64_t p2Rank, CCState &s, int64_t finalp1Rank, bool doubleFlip,
									 std::vector<result> &wins, std::vector<result> &losses)
	{
		r.unrank(p1Rank, p2Rank, s);
		cc.SymmetryFlipVert(s);
		//s.Reverse();
		s.toMove = kMaxPlayer;
		
		//				printf("++%d++\n", p2Rank);
		//				printf("-(A) " ); s.PrintASCII();
		
		int64_t r1, r2;
		assert(s.toMove == kMaxPlayer);
		int64_t theRank = r.rank(s, r1, r2);
		if (groups[r1].symmetricRank == -1)
		{
			// We can look up the children, but we can't store the parent value. Skip.
			return;
		}
		if (data.Get(groups[r1].memoryOffset*memoryMult+r2) != kDraw)
			return;
		
		tResult result = GetValue(cc, s, finalp1Rank, doubleFlip);
		
		switch (result)
		{
			case kWin:
				assert(!"(c) Shouldn't get to this line (wins are imediately propagated)");
				proven++;
				exit(0);
				break;
			case kLoss:
				// Loss symmetrically becomes win at parent
				PropagateWinToParent(cc, s, wins);
				losses.push_back({static_cast<int32_t>(r1), static_cast<int32_t>(r2)});
//				data.Set(groups[r1].memoryOffset*memoryMult+r2, kLoss);
//				groups[r1].changed = true;
//				proven++;
				break;
			case kDraw: // still unknown
				break;
			case kIllegal:
				assert(!"(d) Shouldn't get here");
				break;
		}
	}
	
	void Solver::WriteResult(std::vector<result> &results, tResult valueToWrite)
	{
		lock.lock();
		for (auto &v : results)
		{
//			uint64_t res = data.SetAndTest(groups[v.r1].memoryOffset*memoryMult+v.r2, valueToWrite);
//			if (res == kDraw)
//			{
//				//groups[v.r1].changed = true;
//				open[v.r1] = true;
//				proven++;
//			}
			bool wasSet = data.SetIf(groups[v.r1].memoryOffset*memoryMult+v.r2, valueToWrite, kDraw);
			if (wasSet)
			{
				open[v.r1] = true;
				proven++;
			}
		}
		lock.unlock();
		results.clear();
	}

	void Solver::ThreadMain()
	{
		CCheckers cc;
		CCState s;
		range work;
		std::vector<result> wins(3*CACHE_WRITE_SIZE/2);
		std::vector<result> losses(3*CACHE_WRITE_SIZE/2);
		wins.resize(0);
		losses.resize(0);
		
		// wait to start
		{
			uint8_t tmp;
			startQueue.WaitRemove(tmp);
		}

		while (true)
		{
			workQueue.WaitRemove(work);
			if (work.p1Group == -1) // stop until told to start again
			{
				uint8_t tmp;
				doneQueue.WaitAdd(0);
				startQueue.WaitRemove(tmp);
				continue;
			}

			int64_t p1Rank = work.p1Group;
			for (int64_t p2Rank = work.from; p2Rank < work.to; p2Rank++)
			{
				// Pass #1. These are all the parents that lead directly to the p1Rank as second-player position
				SinglePassInnerLoop(cc, p1Rank, p2Rank, s, p1Rank, false, wins, losses);
				
				// Pass #2. These are all the parents that lead directly to the reversed p1Rank as second-player position
				// These are new states that we have to consider (compared to older solving approaches) that have all
				// their successors in this group
				if (groups[p1Rank].symmetricRank != -1) // There is a symmetric group that leads here
					SinglePassInnerLoop(cc, groups[p1Rank].symmetricRank, p2Rank, s, p1Rank, true, wins, losses);

				if (wins.size() > CACHE_WRITE_SIZE)
					WriteResult(wins, kWin);
				if (losses.size() > CACHE_WRITE_SIZE)
					WriteResult(losses, kLoss);
			}
			WriteResult(wins, kWin);
			WriteResult(losses, kLoss);
		}
	}
	
	void Solver::SinglePass(int numThreads)
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();
		float perc = 0.05f;
		
		if (threads.size() == 0)
		{
			for (int x = 0; x < numThreads; x++)
				threads.push_back(std::thread(&Solver::ThreadMain, this));
		}

		// Tell threads they can start working again
		for (int x = 0; x < numThreads; x++)
		{
			startQueue.WaitAdd(0);
		}

		for (int64_t x = 0; x < order.size(); x++)
		{
			int64_t p1Rank = order[x];
			if ((float)x/order.size() > perc)
			{
				printf("%1.1f%% ", x*100.0/order.size());
				perc += 0.05;
			}
			if (!open[p1Rank])
				//if (!groups[p1Rank].changed)
			{
				continue;
			}
			if (groups[p1Rank].symmetricRank == -1)
			{
				// No states in this group to analyze
				continue;
			}
			lock.lock();
			open[p1Rank] = false;
			lock.unlock();
			//groups[p1Rank].changed = false;
			
			for (int64_t p2Rank = 0; p2Rank < max2; p2Rank+=WORK_SIZE)
			{
				workQueue.WaitAdd({p1Rank, p2Rank, std::min(max2, p2Rank+WORK_SIZE)});
			}
		}
		printf("100%%\n");
		// Tell threads to stop working
		for (int x = 0; x < numThreads; x++)
			workQueue.WaitAdd({-1, -1, -1});
		// Wait for threads to finish
		for (int x = 0; x < numThreads; x++)
		{
			uint8_t tmp;
			doneQueue.WaitRemove(tmp);
		}
	}
	
	// Full size:
	// --> SYMMETRY: 85251690988464 total; 42645604101646 symmetric
	void Solver::BuildData(const char *path, int numThreads)
	{
		printf("-- Starting Parallel Solver (optimized ordering - %d threads) --\n", numThreads);
		printf("-- Ranking: %s --\n", r.name());
		Timer t, total;
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		data.Resize(maxRank);
		data.Clear();
		InitMetaData();
		proven = 0;
		
		total.StartTimer();
		t.StartTimer();
		printf("** Filling in initial states\n");
		Initial(numThreads);
		t.EndTimer();
		printf("Round 0; %llu new; %llu of %llu proven; %1.2fs elapsed\n", proven, proven, symmetricStates, t.GetElapsedTime());
		
		uint64_t oldProven;
		
		CCState start;
		cc.Reset(start);
		int64_t startR1, startR2;
		bool startProven = false;
		r.rank(start, startR1, startR2);
		printf("** Starting Main Loop\n");
		int iteration = 0;
		do {
			iteration++;
			t.StartTimer();
			oldProven = proven;
			SinglePass(numThreads);
			t.EndTimer();
			printf("Round %d; %llu new; %llu of %llu proven; %1.2fs elapsed\n", iteration, proven-oldProven, proven, symmetricStates, t.GetElapsedTime());
			
			if (!startProven && data.Get(groups[startR1].memoryOffset*memoryMult+startR2) != kDraw)
			{
				printf("Start state proven to be %s\n", resultText[data.Get(groups[startR1].memoryOffset*memoryMult+startR2)]);
				startProven = true;
			}
		} while (proven != oldProven);
		total.EndTimer();
		printf("%1.2fs total time elapsed\n", total.EndTimer());
		
		data.Write(GetFileName(path));
		PrintStats();
	}
	
	void Solver::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case ParallelOne::kWin: w++; break;
				case ParallelOne::kLoss: l++; break;
				case ParallelOne::kIllegal: i++; break;
				case ParallelOne::kDraw: d++; break;
			}
		}
		printf("--Cache Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";
	}
	
	const char *Solver::GetFileName(const char *path)
	{
		static std::string s;
		s = path;
		s += "CC-SOLVE-PARALLEL-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += "-";
		s += r.name();
		s += ".dat";
		return s.c_str();
	}
	
	tResult Solver::Translate(int nextPlayer, tResult res) const
	{
		switch (nextPlayer)
		{
			case 0: return res;
			case 1:
			{
				// 	kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
				tResult inv[4] = {kDraw, kWin, kLoss, kIllegal};
				return inv[res];
			}
		}
		assert(false);
		return kIllegal;
	}
	
	tResult Solver::Translate(const CCState &s, tResult res) const
	{
		switch (s.toMove)
		{
			case 0: return res;
			case 1:
			{
				// 	kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
				tResult inv[4] = {kDraw, kWin, kLoss, kIllegal};
				return inv[res];
			}
		}
		assert(false);
		return kIllegal;
	}
	
	tResult Solver::Lookup(const CCState &s) const
	{
		static CCheckers cc;
		int64_t p1, p2;
		uint64_t v = r.rank(s, p1, p2);
		if (groups[p1].symmetricRank == -1)
		{
			CCState tmp = s;
			cc.SymmetryFlipHoriz(tmp);
			v = r.rank(tmp, p1, p2);
			if (groups[p1].symmetricRank == -1)
			{
				printf("Flipped state and still symmetry redundant!\n");
				exit(0);
			}
		}
		return Translate(s, (tResult)data.Get(groups[p1].memoryOffset*memoryMult+p2));
		//		return Translate(s, (tResult)data.Get(v));
	}
	
}


