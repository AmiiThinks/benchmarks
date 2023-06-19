//
//  DBEval.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 5/1/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include "DBEval.h"
#include <assert.h>

DBEval::DBEval(const char *prefix, int firstEntry, int firstPiece)
:db(prefix, firstEntry), prefix(prefix)
{
	this->firstPiece = firstPiece;
}

void DBEval::RootState(const CCState &s)
{
}

bool DBEval::canEval(const CCState &s)
{
	return true;
}

bool DBEval::perfectEval(const CCState &s)
{
	return (s.pieces[0][NUM_PIECES-1] > s.pieces[1][0]);
}


double DBEval::eval(CCheckers *cc, CCState &s, int whichPlayer)
{
	int d1, d2;
	bool l1, l2;
	l1 = GetBackPieceAdvancement(s, 0) >= firstPiece;
	l2 = GetBackPieceAdvancement(s, 1) >= firstPiece;
	if (l1)
	{
		bool res = db.GetDepth(s, 0, d1);
		assert(res == true);
	}
	else {
		d1 = GetDepth(prefix.c_str(), s, 0);
	}
	
	if (l2)
	{
		bool res = db.GetDepth(s, 1, d2);
		assert(res == true);
	}
	else {
		d2 = GetDepth(prefix.c_str(), s, 1);
	}
	
	int moveBonus = (s.toMove==whichPlayer)?1:0;
	
	if (whichPlayer == 0)
		return d2-d1+moveBonus;
	return d1-d2+moveBonus;
}
