//
//  UCB.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/20/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__UCB__
#define __CC_UCT__UCB__

#include <stdio.h>
#include "Player.h"

struct UCBData {
	CCMove *m;
	int count;
	double totalPayoff;
};

class UCB : public Player {
public:
	UCB() {}
	CCMove *GetNextAction(CCheckers *cc, const CCState &s, double &, double timeLimit, int depthLimit = 100000);
	double DoRandomPlayout(CCheckers *cc, CCState s);
	CCMove *GetRandomAction(CCheckers *cc, CCState &s);
	virtual const char *GetName()
	{
		return "UCB1";
	}
	void Reset() {}
private:
	double UCBValue(const UCBData &d, int total);
	int rootPlayer;
	uint64_t nodesExpanded;
};

#endif /* defined(__CC_UCT__UCB__) */
