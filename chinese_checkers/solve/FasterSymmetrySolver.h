//
//  FasterSymmetrySolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 1/5/18.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef FasterSymmetrySolver_h
#define FasterSymmetrySolver_h

#include <stdio.h>

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"

namespace FasterSymmetry {
	
	enum tResult {
		kWin = 2, kLoss = 1, kDraw = 0, kIllegal = 3
	};
		
	static const char *resultText[] = {"Draw", "Loss", "Win", "Illegal"};

	enum {
		kMaxPlayer = 0, kMinPlayer = 1
	};

	enum tMeta {
		kHasNoProvenChildren = 0,
		kHasProvenChildren = 1
	};
	
	struct GroupStructure {
		uint8_t symmetryRedundant;
		uint8_t changed;
		uint8_t handled;
		int32_t symmetricRank;
		int32_t assignedCount;
		int64_t memoryOffset;
	};
	
	class Solver {
	public:
		Solver(const char *path, bool optimizeOrder = false, bool forceBuild = false);
		tResult Lookup(const CCState &s) const;
		void BuildData(const char *path);
		void PrintStats() const;
	private:
		void InitMetaData();
		void Initial();
		
		void DoLoops(CCheckers &cc, int64_t max2, int64_t p1Rank, CCState &s);

		void SinglePass();
		void SinglePassInnerLoop(CCheckers &cc, CCState s, int64_t r1, int64_t finalp1Rank, bool doubleFlip);
		tResult GetValue(CCheckers &cc, const CCState &s, int64_t finalp1Rank, bool doubleFlip);
		void PropagateWinToParent(CCheckers &cc, CCState &s);
		void MarkParents(CCheckers &cc, CCState &s);
		void DoBFS();
		void GetSearchOrder();
		const char *GetFileName(const char *path);
		tResult Translate(const CCState &s, tResult res) const;
		tResult Translate(int nextPlayer, tResult res) const;
		NBitArray<2> data;
		uint64_t proven;
		int64_t symmetricStates;
		stats stat;
		CCPSRank12 r;
		std::vector<GroupStructure> groups;
		std::vector<int8_t> bfs;
		std::vector<int64_t> order;
		bool optimize;
	};
	
}


#endif
