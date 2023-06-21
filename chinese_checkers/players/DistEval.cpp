//
//  DistEval.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/17/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include "DistEval.h"

double DistEval::eval(CCheckers *cc, CCState &s, int whichPlayer)
{
	double dist = 0;
	dist -= cc->goalDistance(s, whichPlayer);
	dist += cc->goalDistance(s, 1-whichPlayer);
	dist += (s.toMove==whichPlayer)?1:0;
	return dist;
}
