//
//  LBDistEval.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/24/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__LBDistEval__
#define __CC_UCT__LBDistEval__

#include <stdio.h>
#include "CCheckers.h"

class LBDistEval {
public:
	void RootState(const CCState &s) {};
	double eval(CCheckers *cc, CCState &s, int whichPlayer);
	int LowerBound(CCheckers *cc, CCState &s, int who);
	const char *GetName() { return "lbdist"; }
private:
	uint8_t dist[17];
};

#endif /* defined(__CC_UCT__LBDistEval__) */
