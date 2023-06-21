//
//  OptimizedSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/17/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef OptimizedSolver_h
#define OptimizedSolver_h

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"

namespace CCOptimizedSolver {
	
	enum tResult {
		kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
	};
	
	enum {
		kMaxPlayer = 0, kMinPlayer = 1
	};
	
	enum tMeta {
		kHasNoProvenChildren = 0,
		kHasProvenChildren = 1
	};
	
	template <typename ranker>
	class OptimizedSolver {
	public:
		OptimizedSolver(const char *path);
		tResult Lookup(const CCState &s) const;
		tResult Lookup(uint64_t rank) const;
		void BuildData(const char *path);
		void PrintStats() const;
	private:
		void Initial();
		void SinglePass();
		tResult GetValue(CCheckers &cc, CCState &s);
		void PropagateWinLossToParent(CCheckers &cc, CCState &s, tResult r);
		void MarkParents(CCheckers &cc, CCState &s);
		const char *GetFileName(const char *path);
		NBitArray<2> data;
		NBitArray<1> meta;
		uint64_t proven;
		ranker r;
	};
	
	template <typename ranker>
	OptimizedSolver<ranker>::OptimizedSolver(const char *path)
	{
		FILE *f = fopen(GetFileName(path), "r");
		if (f == 0)
		{
			BuildData(path);
		}
		else {
			data.Read(f);
		}
	}
	
	template <typename ranker>
	void OptimizedSolver<ranker>::Initial()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		
		for (uint64_t x = 0; x < maxRank; x++)
		{
			r.unrank(x, s);
			if (!cc.Legal(s))
			{
				data.Set(x, kIllegal);
				proven++;
				continue;
			}
			int winner = cc.Winner(s);
			if (winner == 0)
			{
				data.Set(x, kWin);
				if (s.toMove == kMinPlayer)
					PropagateWinLossToParent(cc, s, kWin);
				else
					MarkParents(cc, s);
				proven++;
			}
			else if (winner == 1)
			{
				data.Set(x, kLoss);
				if (s.toMove == kMaxPlayer)
					PropagateWinLossToParent(cc, s, kLoss);
				else
					MarkParents(cc, s);
				proven++;
			}
		}
	}
	
	template <typename ranker>
	tResult OptimizedSolver<ranker>::GetValue(CCheckers &cc, CCState &s)
	{
		CCMove *m = cc.getMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t);
			uint64_t rank = r.rank(s);
			cc.UndoMove(s, t);
			
			switch (data.Get(rank))
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
				case kIllegal: // ignore
					break;
			}
		}
		cc.freeMove(m);
		// all values were set
		if (s.toMove == kMaxPlayer)
			return kLoss;
		return kWin;
	}
	
	template <typename ranker>
	void OptimizedSolver<ranker>::PropagateWinLossToParent(CCheckers &cc, CCState &s, tResult result)
	{
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyReverseMove(s, t);
			uint64_t rank = r.rank(s);
			if (data.Get(rank) == kDraw)
			{
				proven++;
				data.Set(rank, result);
				MarkParents(cc, s);
			}
			cc.UndoReverseMove(s, t);
		}
		cc.freeMove(m);
	}
	
	template <typename ranker>
	void OptimizedSolver<ranker>::MarkParents(CCheckers &cc, CCState &s)
	{
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyReverseMove(s, t);
			uint64_t rank = r.rank(s);
			cc.UndoReverseMove(s, t);
			meta.Set(rank, kHasProvenChildren);
		}
		cc.freeMove(m);
	}
	
	
	template <typename ranker>
	void OptimizedSolver<ranker>::SinglePass()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		
		for (uint64_t x = 0; x < maxRank; x++)
		{
			if (meta.Get(x) != kHasProvenChildren || data.Get(x) != kDraw) // draw functions like unknown
			{
				continue;
			}
			
			r.unrank(x, s);
			
			tResult r = GetValue(cc, s);
			if ((r == kWin && s.toMove == kMinPlayer) || (r == kLoss && s.toMove == kMaxPlayer))
				PropagateWinLossToParent(cc, s, r);
			else if (r != kDraw)
				MarkParents(cc, s);
			
			switch (r)
			{
				case kWin:
					data.Set(x, kWin);
					proven++;
					break;
				case kLoss:
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
	
	
	template <typename ranker>
	void OptimizedSolver<ranker>::BuildData(const char *path)
	{
		printf("-- Starting Optimized solver --\n");
		Timer t, total;
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
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
	
	template <typename ranker>
	void OptimizedSolver<ranker>::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case CCOptimizedSolver::kWin: w++; break;
				case CCOptimizedSolver::kLoss: l++; break;
				case CCOptimizedSolver::kIllegal: i++; break;
				case CCOptimizedSolver::kDraw: d++; break;
			}
		}
		printf("--Optimized Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
	}
	
	template <typename ranker>
	const char *OptimizedSolver<ranker>::GetFileName(const char *path)
	{
		static std::string s;
		s = path;
		s += "CC-SOLVE-OPTIMIZED-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += "-";
		s += r.name();
		s += ".dat";
		return s.c_str();
	}
	
	template <typename ranker>
	tResult OptimizedSolver<ranker>::Lookup(const CCState &s) const
	{
		uint64_t v = r.rank(s);
		return (tResult)data.Get(v);
	}
	
	template <typename ranker>
	tResult OptimizedSolver<ranker>::Lookup(uint64_t rank) const
	{
		return (tResult)data.Get(rank);
	}	
}


#endif /* OptimizedSolver_h */
