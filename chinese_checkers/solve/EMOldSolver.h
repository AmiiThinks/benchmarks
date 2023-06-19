//
//  EMOldSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 2/18/18.
//  Copyright © 2018 NS Software. All rights reserved.
//

#ifndef EMOldSolver_h
#define EMOldSolver_h

#include <stdio.h>

#include <stdio.h>

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"
#include "CCVM.h"

namespace ExternalMemoryOld {
	
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
	
	struct GroupStructure {
		uint8_t symmetryRedundant;
		uint8_t changed;
		uint16_t order;
		int32_t symmetricRank;
	};
	
	class Solver {
	public:
		Solver(const char *path, bool forceBuild = false);
		tResult Lookup(const CCState &s) const;
		void BuildData(const char *path);
		void PrintStats() const;
	private:
		void InitMetaData(const char *path);
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
		CCVM *data;
		uint64_t proven;
		int64_t symmetricStates;
		stats stat;
		CCPSRank12 r;
		std::vector<GroupStructure> groups;
		std::vector<int8_t> bfs;
		std::vector<int64_t> order;
	};
	
}



#endif /* EMOldSolver_hpp */
