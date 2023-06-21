//
//  DBEval.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 5/1/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__DBEval__
#define __CC_UCT__DBEval__

#include <stdio.h>
#include "CCheckers.h"
#include "CCEndgameData.h"

class DBEval {
public:
	DBEval(const char *prefix, int firstEntry, int firstPiece);
	void RootState(const CCState &s);
	double eval(CCheckers *cc, CCState &s, int whichPlayer);
	bool canEval(const CCState &s);
	bool perfectEval(const CCState &s);
	const char *GetName()
	{
		return "db_full";
	}
private:
	CCEndGameData db;
	int firstPiece;
	std::string name;
	std::string prefix;
};

#endif /* defined(__CC_UCT__DBEval__) */
