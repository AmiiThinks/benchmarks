//
//  CacheFriendlySolver.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "CacheFriendlySolver.h"
#include <cassert>
#include "Timer.h"

namespace CCCacheFriendlySolver {
	
	CacheSolver::CacheSolver(const char *path, bool forceBuild)
	{
		if (forceBuild)
			BuildData(path);
		else {
			FILE *f = fopen(GetFileName(path), "r");
			if (f == 0)
			{
				BuildData(path);
			}
			else {
				data.Read(f);
			}
		}
	}

	void CacheSolver::Initial()
	{
		Initial1();
	}
	
	void CacheSolver::Initial1()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();

		//		for (uint64_t x = 0; x < maxRank; x++)
		for (int64_t p1Rank = 0; p1Rank < max1; p1Rank++)
		{
			r.unrankP1(p1Rank, s);

			int startCount = cc.GetNumPiecesInStart(s, 0);
			int goalCount = cc.GetNumPiecesInGoal(s, 0);
			// 1. if no pieces in home, then only one possible goal.
			if (startCount == 0 && goalCount == 0)
			{
				r.unrankP2(0, s);
				stat.unrank++;
				
//				// TODO: we probably don't need this check at all
//				if (!cc.Legal(s))
//				{
//					int64_t x = r.rank(p1Rank, 0);
//					if (data.Get(x) == kDraw)
//					{
//						proven++;
//						data.Set(x, kIllegal);
//						groups[p1Rank].changed = true;
//					}
//					continue;
//				}
				if (cc.Winner(s) == -1)
					continue;
				data.Set(r.rank(p1Rank, 0), kLoss);
				groups[p1Rank].changed = true;
				proven++;
				PropagateWinToParent(cc, s); // This is a win at the parent!
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
//				r.unrank(x, s);
				stat.unrank++;
				if (!cc.Legal(s))
				{
					if (data.Get(x) == kDraw)
					{
						proven++;
						groups[p1Rank].changed = true;
					}
					data.Set(x, kIllegal);
					continue;
				}
				switch (cc.Winner(s))
				{
					case -1: // no winner
						break;
					case 0: // not possible, because it's always player 0's turn
						// Actually, this is possible in one situation.
						assert(!"(0) This isn't possible");
						break;
					case 1:
						//assert(s.toMove == kMaxPlayer);
						//assert(data.Get(x) == kDraw);
						data.Set(x, kLoss);
						proven++;
						groups[p1Rank].changed = true;
						PropagateWinToParent(cc, s); // This is a win at the parent!
						break;
				}
			}
		}
	}

	void CacheSolver::Initial2()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();
		
		//		for (uint64_t x = 0; x < maxRank; x++)
		for (int64_t p1Rank = 0; p1Rank < max1; p1Rank++)
		{
			// Split the ranking
			r.unrankP1(p1Rank, s);
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
				//				r.unrank(x, s);
				stat.unrank++;
				if (!cc.Legal(s))
				{
					if (data.Get(x) == kDraw)
						proven++;
					data.Set(x, kIllegal);
					continue;
				}
				switch (cc.Winner(s))
				{
					case -1: // no winner
						break;
					case 0: // not possible, because it's always player 0's turn
						// Actually, this is possible in one situation.
						assert(!"(0) This isn't possible");
						break;
					case 1:
						//assert(s.toMove == kMaxPlayer);
						//assert(data.Get(x) == kDraw);
						data.Set(x, kLoss);
						proven++;
						PropagateWinToParent(cc, s); // This is a win at the parent!
						break;
				}
			}
		}
	}

	
	void CacheSolver::Initial3()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();
		
		for (uint64_t x = 0; x < maxRank; x++)
		{
			r.unrank(x, s);
			stat.unrank++;
			if (!cc.Legal(s))
			{
				if (data.Get(x) == kDraw)
					proven++;
				data.Set(x, kIllegal);
				continue;
			}
			switch (cc.Winner(s))
			{
				case -1: // no winner
					break;
				case 0: // not possible, because it's always player 0's turn
					// Actually, this is possible in one situation.
					assert(!"(0) This isn't possible");
					break;
				case 1:
					//assert(s.toMove == kMaxPlayer);
					//assert(data.Get(x) == kDraw);
					data.Set(x, kLoss);
					proven++;
					PropagateWinToParent(cc, s); // This is a win at the parent!
					break;
			}
		}
	}

	
	/*
	 * result passed in is the win/loss result for the converted parent of each state
	 */
	void CacheSolver::PropagateWinToParent(CCheckers &cc, CCState &s)
	{
		stat.backwardExpansions++;
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyReverseMove(s, t);
			assert(s.toMove == kMinPlayer);
			int64_t p1Rank, p2Rank;
			uint64_t rank = r.rank(s, p1Rank, p2Rank); // inverts state to first player state
			stat.rank++;
			if (data.Get(rank) == kDraw) // state is currently unknown
			{
				proven++;
				data.Set(rank, kWin);
				groups[p1Rank].changed = true;
			}
			cc.UndoReverseMove(s, t);
		}
		cc.freeMove(m);
	}
	
	/*
	 * Returns the value for the parent, so it has to be flipped.
	 */
	tResult CacheSolver::GetValue(CCheckers &cc, CCState s)
	{
		//assert(s.toMove == kMaxPlayer);
		CCMove *m = cc.getMoves(s);
		stat.forwardExpansions++;
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t); // Now min player to move
			//assert(s.toMove == kMinPlayer);
			uint64_t rank = r.rank(s); // Lookup for max player to move
			stat.rank++;
			tResult val = Translate(s, (tResult)data.Get(rank)); // Invert back to get value for min player
			cc.UndoMove(s, t);
			
			switch (val)
			{
				case kWin:
					if (s.toMove == kMaxPlayer)
					{
						assert(!"(a) Shouldn't get here.\n");
						cc.freeMove(m);
						return kWin;
					}
					break;
				case kLoss:
					if (s.toMove == kMinPlayer)
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
		// all values were set
		//if (s.toMove == kMaxPlayer)
		return kLoss;
		//return kWin;
	}
	
	void CacheSolver::SinglePass()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();
		
		//for (int64_t x = 0; x < max1; x++)
		// This starts from wins and works backwards
		for (int64_t p1Rank = max1-1; p1Rank >= 0; p1Rank--)
		{
			if (!groups[p1Rank].changed)
			{
				continue;
			}
			groups[p1Rank].changed = false;
			for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
			{
				r.unrank(p1Rank, p2Rank, s);
				//s.Reverse();
				cc.SymmetryFlipVert(s);
				s.toMove = 0;

				int64_t r1, r2;
				int64_t theRank = r.rank(s, r1, r2);
				if (data.Get(theRank) != kDraw)
					continue;

				tResult result = GetValue(cc, s);
				
				switch (result)
				{
					case kWin:
						assert(!"(c) Shouldn't get to this line (wins are imediately propagated)");
						data.Set(theRank, kWin);
						proven++;
						exit(0);
						break;
					case kLoss:
						// Loss symmetrically becomes win at parent
						PropagateWinToParent(cc, s);
						data.Set(theRank, kLoss);
						groups[p1Rank].changed = true;
						proven++;
						break;
					case kDraw: // still unknown
						break;
					case kIllegal:
						assert(!"(d) Shouldn't get here");
						break;
				}
			}
		}
	}
	
	
	void CacheSolver::BuildData(const char *path)
	{
		printf("-- Starting Cache solver --\n");
		printf("-- Ranking: %s --\n", r.name());
		Timer t, total;
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		data.Resize(maxRank);
		data.Clear();
		groups.resize(r.getMaxP1Rank());
		for (int x = 0; x < groups.size(); x++)
		{
			groups[x].changed = 0;
		}
		proven = 0;
		
		total.StartTimer();
		t.StartTimer();
		printf("** Filling in initial states\n");
		Initial();
		t.EndTimer();
		printf("Round 0; %llu new; %llu of %llu proven; %1.2f elapsed\n", proven, proven, maxRank, t.GetElapsedTime());
		
		uint64_t oldProven;
		
		printf("** Starting Main Loop\n");
		int iteration = 0;
		do {
			iteration++;
			t.StartTimer();
			oldProven = proven;
			SinglePass();
			t.EndTimer();
			printf("Round %d; %llu new; %llu of %llu proven; %1.2fs elapsed\n", iteration, proven-oldProven, proven, maxRank, t.GetElapsedTime());
		} while (proven != oldProven);
		total.EndTimer();
		printf("%1.2fs total time elapsed\n", total.EndTimer());
		
		data.Write(GetFileName(path));
		PrintStats();
	}
	
	void CacheSolver::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case CCCacheFriendlySolver::kWin: w++; break;
				case CCCacheFriendlySolver::kLoss: l++; break;
				case CCCacheFriendlySolver::kIllegal: i++; break;
				case CCCacheFriendlySolver::kDraw: d++; break;
			}
		}
		printf("--Cache Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";
	}
	
	const char *CacheSolver::GetFileName(const char *path)
	{
		static std::string s;
		s = path;
		s += "CC-SOLVE-CACHE-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += "-";
		s += r.name();
		s += ".dat";
		return s.c_str();
	}
	
	tResult CacheSolver::Translate(const CCState &s, tResult res) const
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
	
	tResult CacheSolver::Lookup(const CCState &s) const
	{
		uint64_t v = r.rank(s);
		return Translate(s, (tResult)data.Get(v));
	}
	
}


