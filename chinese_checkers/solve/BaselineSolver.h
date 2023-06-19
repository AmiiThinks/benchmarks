//
//  BaselineSolver.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/10/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef BaselineSolver_h
#define BaselineSolver_h

#include <stdio.h>
#include "CCheckers.h"
#include "NBitArray.h"

namespace CCBaselineSolver {

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
	
	class BaselineSolver {
	public:
		BaselineSolver(const char *path, bool force);
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
		CCheckers cc_internal;
	};

}

#endif /* BaselineSolver_hpp */
