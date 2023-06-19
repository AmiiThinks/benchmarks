//
//  DBPlusEval.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 5/4/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__DBPlusEval__
#define __CC_UCT__DBPlusEval__

#include <stdio.h>
#include "CCheckers.h"
#include "CCEndgameData.h"
#include <string>
#include "CCUtils.h"
#include <assert.h>

// TODO: create option to allow playout some # of steps to validate distance

template <typename evalFcn>
class DBPlusEval {
public:
	DBPlusEval(evalFcn *f, const char *prefix, int firstEntry, int firstPiece)
	:e(f), db(prefix, firstEntry), firstPiece(firstPiece) { validateDBValue = false; }
	void RootState(const CCState &s) {}
	double eval(CCheckers *cc, CCState &s, int whichPlayer);
	bool canEval(const CCState &s);
	bool perfectEval(const CCState &s);
	int trueDistance(CCheckers *cc, const CCState &s, int who) const;
	const char *GetName()
	{
		name = e->GetName();
		name+="db^";
		name+=std::to_string(firstPiece);
		if (validateDBValue)
			name += "+";
		return name.c_str();
	}
	bool validateDBValue;
private:
	CCEndGameData db;
	evalFcn *e;
	int firstPiece;
	std::string name;
	mutable CCState tmp;
};

template <typename evalFcn>
bool DBPlusEval<evalFcn>::canEval(const CCState &s)
{
	return
	((GetBackPieceAdvancement(s, 0) >= firstPiece) &&
	 (GetBackPieceAdvancement(s, 1) >= firstPiece));
}

template <typename evalFcn>
bool DBPlusEval<evalFcn>::perfectEval(const CCState &s)
{
	return (s.pieces[0][NUM_PIECES-1] > s.pieces[1][0]) &&
	((GetBackPieceAdvancement(s, 0) >= firstPiece) &&
	 (GetBackPieceAdvancement(s, 1) >= firstPiece));
}

template <typename evalFcn>
int DBPlusEval<evalFcn>::trueDistance(CCheckers *cc, const CCState &s, int who) const
{
	int depth;
	int estimate = 0;
	int cnt = 0;
	uint64_t rank = cc->rankPlayer(s, who);
	cc->unrankPlayer(rank, tmp, who);
	CCMove *m = cc->getMovesForward(tmp);
	bool res = db.GetDepth(s, who, depth);
	assert(res == true);
	for (CCMove *t = m; t; t = t->next)
	{
		cc->ApplyMove(tmp, t);
		cnt++;
		int val;
		bool result = db.GetDepth(tmp, who, val);
		assert(result);
		estimate += val;
		cc->UndoMove(tmp, t);
	}
	cc->freeMove(m);

	if (cnt == 0)
		return depth;
	estimate /= cnt;
	if (abs(depth-estimate) > 5)
	{
		depth = depth%15;
		while (abs(depth-estimate) > 5)
			depth += 15;
	}

	return depth;
}


template <typename evalFcn>
double DBPlusEval<evalFcn>::eval(CCheckers *cc, CCState &s, int whichPlayer)
{
	int d1, d2;
	bool l1, l2;
	l1 = GetBackPieceAdvancement(s, 0) >= firstPiece;
	l2 = GetBackPieceAdvancement(s, 1) >= firstPiece;
	if (l1 && l2)
	{
		if (validateDBValue)
		{
			d1 = trueDistance(cc, s, 0);
			d2 = trueDistance(cc, s, 1);
		}
		else {
			bool res = db.GetDepth(s, 0, d1);
			assert(res == true);
			res = db.GetDepth(s, 1, d2);
			assert(res == true);
		}
		int moveBonus = (s.toMove==whichPlayer)?1:0;
		if (whichPlayer == 0)
			return d2-d1+moveBonus;
		return d1-d2+moveBonus;
	}
	else {
		return e->eval(cc, s, whichPlayer);
	}
}


#endif /* defined(__CC_UCT__DBPlusEval__) */
