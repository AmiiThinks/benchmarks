//
//  LBDistEval.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/24/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include "LBDistEval.h"
#include <string.h>

double LBDistEval::eval(CCheckers *cc, CCState &s, int whichPlayer)
{
	double dist = 0;
	dist -= LowerBound(cc, s, whichPlayer);
	dist += LowerBound(cc, s, 1-whichPlayer);
	return dist;
}

int LBDistEval::LowerBound(CCheckers *cc, CCState &s, int who)
{
	int inPlace = 0;
	int moves1 = 0;
	int common = 0;
	memset(dist, 0, 17);
	for (int x = 0; x < NUM_PIECES; x++)
	{
		int next = cc->distance(s.pieces[who][x], cc->getGoal(who));
		dist[next]++;
	}
	for (int x = 0; x < 17 && inPlace != NUM_PIECES; x++)
	{
		if (x < 4)
		{
			inPlace += dist[x];
		}
		else {
			if (dist[x] == 0)
			{
				// at least one move to fill in empty row for later jumping
				moves1++;
			}
			else {
				common += dist[x]*x;
				inPlace += dist[x];
			}
		}
	}
	return common;//+moves1;
}
