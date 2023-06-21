//
//  SimpleSolve.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 7/23/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include "SimpleSolver.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "CCheckers.h"

void SimpleSolveGame(std::vector<uint8_t> &wins)
{
	int provingPlayer = 0;
	CCheckers cc;
	CCState s;
	
	cc.Reset(s);
	uint64_t root = cc.rank(s);
	uint64_t maxRank = cc.getMaxRank();
	wins.resize(maxRank);
	uint64_t winningStates = 0;
	uint64_t illegalStates = 0;
	uint64_t legalStates = 0;
	printf("Finding won positions for player %d\n", provingPlayer);
	float perc = 0;
	for (uint64_t val = 0; val < maxRank; val++)
	{
		wins[val] = 0;

		if (100.0*val/maxRank >= perc)
		{
			perc+=5;
			std::cout << val << " of " << maxRank << " " << 100.0*val/maxRank << "% complete " << 
			legalStates << " legal " << winningStates << " winning " << illegalStates << " illegal" << std::endl;
		}
		if (cc.unrank(val, s))
		{
			legalStates++;
			if (cc.Done(s))
			{
				if (cc.Winner(s) == provingPlayer)
				{
					if (s.toMove == 1-provingPlayer)
					{
//						s.PrintASCII();
						wins[val] = 1;
						winningStates++;
					}
					else {
						illegalStates++;
					}
				}
			}
		}
		else {
		}
	}
	std::cout << maxRank << " of " << maxRank << " 100% complete " << 
	legalStates << " legal " << winningStates << " winning " << illegalStates << " illegal" << std::endl;

	printf("%lld states unranked; %lld were winning; %lld were tech. illegal\n",
		   legalStates, winningStates, illegalStates);
	int changed = 0;
	int round = 0;
	do {
		changed = 0;
		if (wins[root])
		{
			printf("Win proven for player %d\n", provingPlayer);
			//exit(0);
		}
		// propagating wins
		for (uint64_t val = 0; val < maxRank; val++)
		{
			if (wins[val])
			{
				//printf("%d winning states left\n", --winningStates);
			}
			else if (cc.unrank(val, s))
			{
//				printf("State: "); s.PrintASCII();

				if (cc.Winner(s) == (1-provingPlayer))
					continue;

				CCMove *m = cc.getMoves(s);
				bool done = false;
				bool proverToMove = (s.toMove==provingPlayer);
				for (CCMove *t = m; t && !done; t = t->next)
				{
					cc.ApplyMove(s, t);
					uint64_t succ = cc.rank(s);
					if ((proverToMove) && wins[succ])
					{
						if (val == root)
						{
							printf("Win proven for player %d\n", provingPlayer);
							//return;
						}
						wins[val] = round+2;
						done = true;
						changed++;
					}
					if ((!proverToMove) && (!wins[succ]))
					{
						done = true;
					}
					cc.UndoMove(s, t);
				}
				if (!done && !proverToMove)
				{
					wins[val] = round+2;
					changed++;
				}
				cc.freeMove(m);
			}
		}
		printf("round %d; %d changed\n", round++, changed);
	} while (changed != 0);
//	printf("Win not proven\n");
}

