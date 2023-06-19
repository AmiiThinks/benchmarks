//
//  DistDBEval.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/22/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__DistDBEval__
#define __CC_UCT__DistDBEval__

#include <stdio.h>
#include "CCheckers.h"
#include "CCEndgameData.h"
#include <string>

enum lookupType {
	kDBOnly,
	kDBAndDist,// can mix DB and dist
	kDBXorDist // DB or dist, but not both
};

class DistDBEval {
public:
	DistDBEval(const char *prefix, int firstEntry, int firstPiece);
	void RootState(const CCState &s);
	double eval(CCheckers *cc, CCState &s, int whichPlayer);
	bool canEval(const CCState &s);
	bool perfectEval(const CCState &s);
	const char *GetName()
	{
		switch (lookup) {
			case kDBOnly: name = "db"; break;
			case kDBXorDist: name = "dist^db"; break;
			case kDBAndDist: name = "dist+db"; break;
		}
	    name += std::to_string(firstPiece);
		return name.c_str();
    }
	lookupType lookup;
private:
	CCEndGameData db;
	int firstPiece;
	std::string name;
};

#endif /* defined(__CC_UCT__DistDBEval__) */
