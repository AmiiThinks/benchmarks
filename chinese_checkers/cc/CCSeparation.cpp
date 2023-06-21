//
//  CCSeparation.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 8/17/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include "CCSeparation.h"


tSeparationValue TestSeparationWin(CCheckers &cc, CCState &s, std::vector<uint8_t> &d, int provingPlayer)
{
	return kUnknown;
	
//	int maxr=-1;
// int minr=100;
//	// crude separation test; might be falliable
//	for (int x = 0; x < NUM_PIECES; x++)
//	{
//		int tmpx, tmpy;
//		cc.toxy(s.pieces[1][x], tmpx, tmpy);
//		if (tmpy > maxr)
//			maxr = tmpy;
//		cc.toxy(s.pieces[0][x], tmpx, tmpy);
//		if (tmpy < minr)
//			minr = tmpy;
//	}
//	if (maxr > minr) //if (maxr >= minr)
//		return kUnknown;
//	
//	CCState t = s;
//	t.Reverse();
//	int d0 = d[cc.rankPlayer(t, 0)];
//	int d1 = d[cc.rankPlayer(s, 1)];
//	if (provingPlayer == 0)
//	{
//		if (d0 < d1)
//		{
////			printf("%d for 0; %d for 1\n", d0, d1);
//			return kProvenWin;
//		}
//		if (s.toMove == 0 && d0 == d1)
//		{
////			printf("%d for 0; %d for 1\n", d0, d1);
//			return kProvenWin;
//		}
//		return kProvenLoss;
//	}
//	else {
//		if (d1 < d0)
//			return kProvenWin;
//		if (s.toMove == 1 && d0 == d1)
//			return kProvenWin;
//		return kProvenLoss;		
//	}
}
