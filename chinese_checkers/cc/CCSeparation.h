//
//  CCSeparation.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 8/17/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include <vector>
#include "CCheckers.h"

enum tSeparationValue {
	kProvenWin,
	kProvenLoss,
	kUnknown
};

tSeparationValue TestSeparationWin(CCheckers &cc, CCState &s, std::vector<uint8_t> &d, int provingPlayer);
