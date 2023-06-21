//
//  CCEndgameData.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/22/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__CCEndgameData__
#define __CC_UCT__CCEndgameData__

#include <stdio.h>
#include "CCUtils.h"

class CCEndGameData {
public:
	CCEndGameData(const char *prefix, int firstTwoPieceRank);
	~CCEndGameData();
	bool GetRawDepth(const CCState &s, int who, int &depth) const;
	bool GetDepth(const CCState &s, int who, int &depth) const;
	int LowerBound(const CCState &s, int who) const;
private:
	std::vector<uint8_t*> rawData;
	mutable uint8_t dist[17];
	CCheckers cc;
};

#endif /* defined(__CC_UCT__CCEndgameData__) */
