//
//  CacheFriendlySolver.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "FullSymmetrySolver.h"
#include <cassert>
#include "Timer.h"

namespace FullSymmetry {
	static const char *resultText[] = {"Draw", "Loss", "Win", "Illegal"};

	Solver::Solver(const char *path, bool optimizeOrder, bool forceBuild)
	:optimize(optimizeOrder)
	{
		if (NUM_PIECES == 2 || NUM_PIECES == 5)
		{
			printf("%d pieces not symmetric; aborting\n", NUM_PIECES);
			exit(0);
		}
		if (forceBuild)
		{
			BuildData(path);
		}
		else {
			FILE *f = fopen(GetFileName(path), "r");
			if (f == 0)
			{
				BuildData(path);
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
		groups.resize(r.getMaxP1Rank());
		
		// Note that in single-player CC we have to (maybe?) generate
		// the children of both the regular and flipped states
		
		int64_t totalStates = 0;
		symmetricStates = 0;
		for (int32_t x = 0; x < groups.size(); x++)
			groups[x].symmetricRank = -1;
		for (int32_t x = 0; x < groups.size(); x++)
		{
			groups[x].changed = 0;
			groups[x].handled = false;
			groups[x].assignedCount = 0;
			r.unrank(x, 0, s);
			CCState sym;
			cc.FlipPlayer(s, sym, 0);

			// Since we are only going up to 7x7 (6 piece) CC, we can narrow to 32 bits here
			int32_t otherRank = static_cast<int32_t>(r.rankP1(sym));
			if (otherRank < x)
			{
				groups[x].symmetryRedundant = true;
				groups[x].symmetricRank = otherRank;
				groups[otherRank].symmetricRank = x;
			}
			else {
				groups[x].memoryOffset = memoryOffset;
				memoryOffset += r.getMaxP2Rank();
				groups[x].symmetryRedundant = false;
				//printf("[%d] !Symmetric: ", x); s.PrintASCII();
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
		if (optimize)
		{
			int d = 0;
			while (order.size() != groups.size())
			{
				for (int x = bfs.size()-1; x >= 0; x--)
				{
					if (bfs[x] == d)
						order.push_back(x);
				}
				d++;
			}
		}
		else {
			for (int x = groups.size()-1; x >= 0; x--)
				order.push_back(x);
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
	
	void Solver::Initial()
	{
		CCheckers cc;
		CCState s, tmp;
		uint64_t maxRank = r.getMaxRank();
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();

		//		for (uint64_t x = 0; x < maxRank; x++)
		for (int64_t p1Rank = 0; p1Rank < max1; p1Rank++)
		{
			if (groups[p1Rank].symmetryRedundant)
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
				data.Set(groups[p1Rank].memoryOffset, kLoss);
				groups[p1Rank].assignedCount++;
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
				stat.unrank++;
				if (!cc.Legal(s))
				{
					if (data.Get(groups[p1Rank].memoryOffset+p2Rank) == kDraw)
					{
						proven++;
						groups[p1Rank].assignedCount++;
						groups[p1Rank].changed = true;
					}
					data.Set(groups[p1Rank].memoryOffset+p2Rank, kIllegal);
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
						data.Set(groups[p1Rank].memoryOffset+p2Rank, kLoss);
						proven++;
						groups[p1Rank].assignedCount++;
						groups[p1Rank].changed = true;
						tmp = s;
						PropagateWinToParent(cc, tmp); // This is a win at the parent!
						break;
				}
			}
		}
	}
	
	/*
	 * Note that s will be modified in this function
	 */
	void Solver::PropagateWinToParent(CCheckers &cc, CCState &s)
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
			if (!groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(s);

				if (data.Get(groups[p1Rank].memoryOffset+p2Rank) == kDraw)
				{
					proven++;
					data.Set(groups[p1Rank].memoryOffset+p2Rank, kWin);
					groups[p1Rank].assignedCount++;
					groups[p1Rank].changed = true;
				}
			}

			// Only flip pieces
			cc.SymmetryFlipHoriz_PO(s, tmp);

			p1Rank = r.rankP1(tmp);

			stat.rank++;
			if (!groups[p1Rank].symmetryRedundant)
			{
				p2Rank = r.rankP2(tmp);

				if (data.Get(groups[p1Rank].memoryOffset+p2Rank) == kDraw)
				{
					proven++;
					data.Set(groups[p1Rank].memoryOffset+p2Rank, kWin);
					groups[p1Rank].assignedCount++;
					groups[p1Rank].changed = true;
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
			val = Translate(kMaxPlayer, (tResult)data.Get(groups[p1Rank].memoryOffset+p2Rank)); // Invert back to get value for min player
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
	
	void Solver::SinglePassInnerLoop(CCheckers &cc, int64_t p1Rank, int64_t p2Rank, CCState &s, int64_t finalp1Rank, bool doubleFlip)
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
		if (groups[r1].symmetryRedundant)
		{
			// We can look up the children, but we can't store the parent value. Skip.
			return;
		}
		if (data.Get(groups[r1].memoryOffset+r2) != kDraw)
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
				PropagateWinToParent(cc, s);
				data.Set(groups[r1].memoryOffset+r2, kLoss);
				groups[r1].assignedCount++;
				groups[r1].changed = true;
				proven++;
				break;
			case kDraw: // still unknown
				break;
			case kIllegal:
				assert(!"(d) Shouldn't get here");
				break;
		}
	}
	
	void Solver::SinglePass()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		
		int64_t max1 = r.getMaxP1Rank();
		int64_t max2 = r.getMaxP2Rank();
		float perc = 0.05f;
		
		for (int64_t x = 0; x < order.size(); x++)
		{
			int64_t p1Rank = order[x];
			if ((float)x/order.size() > perc)
			{
				printf("%1.1f%% ", x*100.0/order.size());
				perc += 0.05;
			}
			if (!groups[p1Rank].changed)
			{
				continue;
			}
			if (groups[p1Rank].symmetryRedundant)
			{
				// No states in this group to analyze
				continue;
			}
			groups[p1Rank].changed = false;
			
			for (int64_t p2Rank = 0; p2Rank < max2; p2Rank++)
			{
				// Pass #1. These are all the parents that lead directly to the p1Rank as second-player position
				SinglePassInnerLoop(cc, p1Rank, p2Rank, s, p1Rank, false);

				// Pass #2. These are all the parents that lead directly to the reversed p1Rank as second-player position
				// These are new states that we have to consider (compared to older solving approaches) that have all
				// their successors in this group
				if (groups[p1Rank].symmetricRank != -1) // There is a symmetric group that leads here
					SinglePassInnerLoop(cc, groups[p1Rank].symmetricRank, p2Rank, s, p1Rank, true);
			}
		}
		printf("\n");
	}
	
	// Full size:
	// --> SYMMETRY: 85251690988464 total; 42645604101646 symmetric
	void Solver::BuildData(const char *path)
	{
		if (optimize)
			printf("-- Starting Full Symmetry Solver solver (optimized ordering) --\n");
		else
			printf("-- Starting Full Symmetry Solver solver (no order optimization) --\n");
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
		Initial();
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
			SinglePass();
			t.EndTimer();
			printf("Round %d; %llu new; %llu of %llu proven; %1.2fs elapsed\n", iteration, proven-oldProven, proven, symmetricStates, t.GetElapsedTime());

			if (!startProven && data.Get(groups[startR1].memoryOffset+startR2) != kDraw)
			{
				printf("Start state proven to be %s\n", resultText[data.Get(groups[startR1].memoryOffset+startR2)]);
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
				case FullSymmetry::kWin: w++; break;
				case FullSymmetry::kLoss: l++; break;
				case FullSymmetry::kIllegal: i++; break;
				case FullSymmetry::kDraw: d++; break;
			}
		}
		printf("--Cache Data Summary [raw data]--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);

		w = 0;
		l = 0;
		d = 0;
		i = 0;
		
		for (uint64_t x = 0; x < r.getMaxP1Rank(); x++)
		{
			if (groups[x].symmetryRedundant)
				continue;
			
			bool sym = false;
			if (groups[x].symmetricRank != -1)
				sym = true;
			for (uint64_t y = 0; y < r.getMaxP2Rank(); y++)
			{
				switch (data.Get(groups[x].memoryOffset+y))
				{
					case FullSymmetry::kWin: w+=(sym?2:1); break;
					case FullSymmetry::kLoss: l+=(sym?2:1); break;
					case FullSymmetry::kIllegal: i+=(sym?2:1); break;
					case FullSymmetry::kDraw: d+=(sym?2:1); break;
				}
			}
		}
		uint64_t tmp = w;
		w += l;
		l+=tmp;
		d*=2;
		i*=2;
		
		printf("--Cache Data Summary [normalized to full data without symmetry]--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";
	}
	
	const char *Solver::GetFileName(const char *path)
	{
		static std::string s;
		s = path;
		s += "CC-SOLVE-FULLSYM-";
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
		if (groups[p1].symmetryRedundant)
		{
			CCState tmp = s;
			cc.SymmetryFlipHoriz(tmp);
			v = r.rank(tmp, p1, p2);
			if (groups[p1].symmetryRedundant)
			{
				printf("Flipped state and still symmetry redundant!\n");
				exit(0);
			}
		}
		return Translate(s, (tResult)data.Get(groups[p1].memoryOffset+p2));
//		return Translate(s, (tResult)data.Get(v));
	}
	
}


