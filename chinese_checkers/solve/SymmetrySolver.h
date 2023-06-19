//
//  SymmetrySolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/25/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef SymmetrySolver_h
#define SymmetrySolver_h

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"

namespace CCSymmetrySolver {
	
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
	
	template <typename ranker, bool markParents = true>
	class SymmetrySolver {
	public:
		SymmetrySolver(const char *path, bool forceBuild = false);
		tResult Lookup(const CCState &s) const;
		void BuildData(const char *path);
		void PrintStats() const;
	private:
		void Initial();
		void SinglePass();
		tResult Translate(const CCState &s, tResult res) const;
		tResult GetValue(CCheckers &cc, CCState &s);
		void PropagateWinToParent(CCheckers &cc, CCState &s);
		void MarkParents(CCheckers &cc, CCState &s);
		const char *GetFileName(const char *path);
		NBitArray<2> data;
		NBitArray<1> meta;
		uint64_t proven;
		ranker r;
		stats stat;
	};
	
	template <typename ranker, bool markParents>
	SymmetrySolver<ranker, markParents>::SymmetrySolver(const char *path, bool forceBuild)
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
	
	template <typename ranker, bool markParents>
	void SymmetrySolver<ranker, markParents>::Initial()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		
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
	template <typename ranker, bool markParents>
	void SymmetrySolver<ranker, markParents>::PropagateWinToParent(CCheckers &cc, CCState &s)
	{
		stat.backwardExpansions++;
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyReverseMove(s, t);
			//assert(s.toMove == kMinPlayer);
			uint64_t rank = r.rank(s); // inverts state to first player state
			stat.rank++;
			if (data.Get(rank) == kDraw) // state is currently unknown
			{
				proven++;
				data.Set(rank, kWin);
				MarkParents(cc, s);
			}
			cc.UndoReverseMove(s, t);
		}
		cc.freeMove(m);
	}
	
	template <typename ranker, bool markParents>
	void SymmetrySolver<ranker, markParents>::MarkParents(CCheckers &cc, CCState &s)
	{
		if (!markParents)
			return;
		CCMove *m = cc.getReverseMoves(s);
		stat.backwardExpansions++;
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyReverseMove(s, t);
			uint64_t rank = r.rank(s);
			stat.rank++;
			cc.UndoReverseMove(s, t);
			meta.Set(rank, kHasProvenChildren);
		}
		cc.freeMove(m);
	}
	/*
	 * Returns the value for the parent, so it has to be flipped.
	 */
	template <typename ranker, bool markParents>
	tResult SymmetrySolver<ranker, markParents>::GetValue(CCheckers &cc, CCState &s)
	{
		//assert(s.toMove == kMaxPlayer);
		CCMove *m = cc.getMoves(s);
		stat.forwardExpansions++;
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t); // Now min player to move
//			assert(s.toMove == kMinPlayer);
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
	
	template <typename ranker, bool markParents>
	void SymmetrySolver<ranker, markParents>::SinglePass()
	{
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		
		for (uint64_t x = 0; x < maxRank; x++)
		//for (int64_t x = maxRank-1; x >= 0; x--)
		{
			//if (meta.Get(x) != kHasProvenChildren || data.Get(x) != kDraw) // draw functions like unknown
			//if (data.Get(x) != kDraw) // draw functions like unknown
			if (!
				(data.Get(x) == kDraw && (!markParents || meta.Get(x) == kHasProvenChildren))
				)
			{
				continue;
			}
			
			r.unrank(x, s);
			stat.unrank++;
			assert(s.toMove == kMaxPlayer);
			
			tResult result = GetValue(cc, s);
			
			switch (result)
			{
				case kWin:
					assert(!"(c) Shouldn't get to this line (wins are imediately propagated)");
					MarkParents(cc, s);
					data.Set(x, kWin);
					proven++;
					exit(0);
					break;
				case kLoss:
					// Loss symmetrically becomes win at parent
					PropagateWinToParent(cc, s);
					data.Set(x, kLoss);
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
	
	
	template <typename ranker, bool markParents>
	void SymmetrySolver<ranker, markParents>::BuildData(const char *path)
	{
		printf("-- Starting Symmetry solver --\n");
		printf("-- Ranking: %s --\n", r.name());
		Timer t, total;
		CCheckers cc;
		CCState s;
		uint64_t maxRank = r.getMaxRank();
		data.Resize(maxRank);
		data.Clear();
		if (markParents)
		{
			meta.Resize(maxRank);
			meta.Clear();
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
		// Clear meta data
		meta.Resize(0);
		PrintStats();
	}
	
	template <typename ranker, bool markParents>
	void SymmetrySolver<ranker, markParents>::PrintStats() const
	{
		uint64_t w = 0, l = 0, d = 0, i = 0;
		for (uint64_t x = 0; x < data.Size(); x++)
		{
			switch (data.Get(x))
			{
				case CCSymmetrySolver::kWin: w++; break;
				case CCSymmetrySolver::kLoss: l++; break;
				case CCSymmetrySolver::kIllegal: i++; break;
				case CCSymmetrySolver::kDraw: d++; break;
			}
		}
		printf("--Symmetry Data Summary--\n");
		printf("%llu wins\n%llu losses\n%llu draws\n%llu illegal\n", w, l, d, i);
		std::cout << stat << "\n";
	}
	
	template <typename ranker, bool markParents>
	const char *SymmetrySolver<ranker, markParents>::GetFileName(const char *path)
	{
		static std::string s;
		s = path;
		s += "CC-SOLVE-SYM-";
		s += std::to_string(NUM_SPOTS);
		s += "-";
		s += std::to_string(NUM_PIECES);
		s += "-";
		s += r.name();
		s += ".dat";
		return s.c_str();
	}

	template <typename ranker, bool markParents>
	tResult SymmetrySolver<ranker, markParents>::Translate(const CCState &s, tResult res) const
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
	
	template <typename ranker, bool markParents>
	tResult SymmetrySolver<ranker, markParents>::Lookup(const CCState &s) const
	{
		uint64_t v = r.rank(s);
		return Translate(s, (tResult)data.Get(v));
	}
}

#endif /* SymmetrySolver_h */
