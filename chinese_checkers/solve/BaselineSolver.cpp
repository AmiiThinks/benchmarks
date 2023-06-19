//
//  BaselineSolver.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/10/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "BaselineSolver.h"
#include <string>
#include <cassert>
#include "Timer.h"

namespace CCBaselineSolver {
	
	BaselineSolver::BaselineSolver(const char *path, bool force)
	{
		FILE *f = fopen(GetFileName(path), "r");
		if (f == 0 || force)
		{
			BuildData(path);
		}
		else {
			data.Read(f);
			PrintStats();
		}
	}
	
	void BaselineSolver::Initial()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = cc.getMaxRank();
		
		for (uint64_t x = 0; x < maxRank; x++)
		{
			cc.unrank(x, s);
			if (!cc.Legal(s))
			{
				data.Set(x, kIllegal);
				proven++;
			}
		}
		printf("--> %llu ILLEGAL\n", proven);
		for (uint64_t x = 0; x < maxRank; x++)
		{
			if (data.Get(x) == kIllegal)
				continue;

			cc.unrank(x, s);

			int winner = cc.Winner(s);
			if (winner == 0)
			{
				if (data.Get(x) == kDraw)
					proven++;
				data.Set(x, kWin);
				if (s.toMove == kMinPlayer)
					PropagateWinLossToParent(cc, s, kWin);
				else
					MarkParents(cc, s);
			}
			else if (winner == 1)
			{
				if (data.Get(x) == kDraw)
					proven++;
				data.Set(x, kLoss);
				if (s.toMove == kMaxPlayer)
					PropagateWinLossToParent(cc, s, kLoss);
				else
					MarkParents(cc, s);
			}
		}
	}
	
	tResult BaselineSolver::GetValue(CCheckers &cc, CCState &s)
	{
		CCMove *m = cc.getMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t);
			uint64_t r = cc.rank(s);
			cc.UndoMove(s, t);
			
			switch (data.Get(r))
			{
				case kWin:
					if (s.toMove == kMaxPlayer)
					{
						cc.freeMove(m);
						printf("(a) Shouldn't get here.\n");
						return kWin;
					}
					break;
				case kLoss:
					if (s.toMove == kMinPlayer)
					{
						cc.freeMove(m);
						printf("(b) Shouldn't get here.\n");
						return kLoss;
					}
					break;
				case kDraw: // still unknown
					cc.freeMove(m);
					return kDraw;
					break;
				case kIllegal: // ignore - illegal to move to an illegal state
					break;
			}
		}
		cc.freeMove(m);
		// all values were set
		if (s.toMove == kMaxPlayer)
			return kLoss;
		return kWin;
	}
	
	// This state has a parent
	// Parent: [0] 1 1 1 1 1 2 0 0 2 0 2 0 2 1 2 2

	void BaselineSolver::PropagateWinLossToParent(CCheckers &cc, CCState &s, tResult r)
	{
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.UndoMove(s, t);
			uint64_t rank = cc.rank(s);
			if (data.Get(rank) == kDraw)
			{
				proven++;
				data.Set(rank, r);
				MarkParents(cc, s);
			}
			cc.ApplyMove(s, t);
		}
		cc.freeMove(m);
	}
	
	void BaselineSolver::MarkParents(CCheckers &cc, CCState &s)
	{
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.UndoMove(s, t);
			uint64_t rank = cc.rank(s);
			cc.ApplyMove(s, t);
			meta.Set(rank, kHasProvenChildren);
		}
		cc.freeMove(m);
	}

	
	void BaselineSolver::SinglePass()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = cc.getMaxRank();
		
		for (uint64_t x = 0; x < maxRank; x++)
		{
			if (meta.Get(x) != kHasProvenChildren || data.Get(x) != kDraw) // draw functions like unknown
			{
				continue;
			}
			
			cc.unrank(x, s);
			
			tResult r = GetValue(cc, s);
			if ((r == kWin && s.toMove == kMinPlayer) || (r == kLoss && s.toMove == kMaxPlayer))
				PropagateWinLossToParent(cc, s, r);
			else if (r != kDraw)
				MarkParents(cc, s);
			
			switch (r)
			{
				case kWin:
					assert(data.Get(x) == kDraw);
					data.Set(x, kWin);
					proven++;
					break;
				case kLoss:
					assert(data.Get(x) == kDraw);
					data.Set(x, kLoss);
					proven++;
					break;
				case kDraw: // still unknown
					break;
				case kIllegal:
					assert(!"Shouldn't get here");
					break;
			}
		}
	}
	
	
	void BaselineSolver::BuildData(const char *path)
	{
		printf("-- Starting baseline solver --\n");
		Timer t, total;
		CCheckers cc;
		CCState s;
		uint64_t maxRank = cc.getMaxRank();
		data.Resize(maxRank);
		data.Clear();
		meta.Resize(maxRank);
		meta.Clear();
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
		// Clear meta data
		meta.Resize(0);
		PrintStats();
	}

	void BaselineSolver::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case CCBaselineSolver::kWin: w++; break;
				case CCBaselineSolver::kLoss: l++; break;
				case CCBaselineSolver::kIllegal: i++; break;
				case CCBaselineSolver::kDraw: d++; break;
			}
		}
		printf("--Baseline Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
	}

	const char *BaselineSolver::GetFileName(const char *path)
	{
		static std::string s;
		s = path;
		s += "CC-SOLVE-BASELINE-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += ".dat";
		return s.c_str();
	}
	
	tResult BaselineSolver::Lookup(const CCState &s) const
	{
		uint64_t v = cc_internal.rank(s);
		return (tResult)data.Get(v);
	}

	tResult BaselineSolver::Lookup(uint64_t rank) const
	{
		return (tResult)data.Get(rank);
	}

}
