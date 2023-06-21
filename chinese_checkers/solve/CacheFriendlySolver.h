//
//  CacheFriendlySolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 11/5/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef CacheFriendlySolver_h
#define CacheFriendlySolver_h

#include <stdio.h>

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"
#include "SolveStats.h"
#include "CCRankings.h"

namespace CCCacheFriendlySolver {
	
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
	};
	
	class CacheSolver {
	public:
		CacheSolver(const char *path, bool forceBuild = false);
		tResult Lookup(const CCState &s) const;
		void BuildData(const char *path);
		void PrintStats() const;
	private:
		void Initial();
		void Initial1();
		void Initial2();
		void Initial3();
		void SinglePass();
		tResult Translate(const CCState &s, tResult res) const;
		tResult GetValue(CCheckers &cc, CCState s);
		void PropagateWinToParent(CCheckers &cc, CCState &s);
		void MarkParents(CCheckers &cc, CCState &s);
		const char *GetFileName(const char *path);
		NBitArray<2> data;
		uint64_t proven;
		stats stat;
		CCPSRank12 r;
		std::vector<GroupStructure> groups;
	};
	
}


#endif /* CacheFriendlySolver_h */
