//
//  DistEval.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/17/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__DistEval__
#define __CC_UCT__DistEval__

#include <stdio.h>
#include "CCheckers.h"

class DistEval {
public:
	void RootState(const CCState &s) {};
	double eval(CCheckers *cc, CCState &s, int whichPlayer);
	bool canEval(const CCState &) { return true; }
	bool perfectEval(const CCState &s) { return false; }
	const char *GetName() { return "dist"; }
};

#endif /* defined(__CC_UCT__DistEval__) */
