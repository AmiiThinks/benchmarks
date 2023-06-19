//
//  DistDBEval.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/22/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include "DistDBEval.h"
#include "CCUtils.h"
#include <assert.h>

DistDBEval::DistDBEval(const char *prefix, int firstEntry, int firstPiece)
:db(prefix, firstEntry)
{
	lookup = kDBAndDist;
	this->firstPiece = firstPiece;
}

void DistDBEval::RootState(const CCState &s)
{
}

bool DistDBEval::canEval(const CCState &s)
{
	return (lookup != kDBOnly) ||
	((GetBackPieceAdvancement(s, 0) >= firstPiece) &&
	(GetBackPieceAdvancement(s, 1) >= firstPiece));
}

bool DistDBEval::perfectEval(const CCState &s)
{
	return (s.pieces[0][NUM_PIECES-1] > s.pieces[1][0]) &&
	((GetBackPieceAdvancement(s, 0) >= firstPiece) &&
	 (GetBackPieceAdvancement(s, 1) >= firstPiece));
}


double DistDBEval::eval(CCheckers *cc, CCState &s, int whichPlayer)
{
	int d1, d2;
	bool l1, l2, both;
	l1 = GetBackPieceAdvancement(s, 0) >= firstPiece;
	l2 = GetBackPieceAdvancement(s, 1) >= firstPiece;
	both = l1&&l2;
	if ((both && (lookup == kDBXorDist)) || (l1 && (lookup == kDBAndDist)))
	{
		bool res = db.GetDepth(s, 0, d1);
//		if (s.toMove == whichPlayer)
//			d1--;
		assert(res == true);
	}
	else {
		d1 = cc->goalDistance(s, 0);
	}
	
	if ((both && (lookup == kDBXorDist)) || (l2 && (lookup == kDBAndDist)))
	{
		bool res = db.GetDepth(s, 1, d2);
//		if (s.toMove == whichPlayer)
//			d2--;
		assert(res == true);
	}
	else {
		d2 = cc->goalDistance(s, 1);
	}
	
	int moveBonus = (s.toMove==whichPlayer)?1:0;
	
	if (whichPlayer == 0)
		return d2-d1+moveBonus;
	return d1-d2+moveBonus;
}
