//
//  LegacyTests.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/25/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "LegacyTests.h"
#include <vector>
#include "CCheckers.h"
#include "BackwardSolver.h"
#include "ForwardSolver.h"

void ExtractProof(std::vector<int8_t> &data, int64_t rank)
{
	CCheckers cc;
	CCState s;
	
	cc.Reset(s);
	uint64_t root = cc.rank(s);
	
	while (rank != root)
	{
		cc.unrank(rank, s);
		s.Print();
		printf("This state at depth %d\n", data[rank]);
		CCMove *m = cc.getMoves(s);
		bool success = false;
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t);
			int64_t r = cc.rank(s);
			cc.UndoMove(s, t);
			printf("--Child at depth %d :", data[r]);
			t->Print(0);
			printf("\n");
			if (data[r] == data[rank]-1)
			{
				rank = r;
				success = true;
				//break;
			}
		}
		if (!success)
		{
			printf("FAILURE!\n");
			break;
		}
		cc.freeMove(m);
	}
}

void TestConfigParameters()
{
	// Algorithm:
	//  * Forward
	//    - Strict depth
	//    - Permissive Depth+Repair
	//  * Hybrid
	//    - Strict depth
	//    - Permissive Depth+Repair
	//  * Backward
	//    - Expand all
	//    - Expand at depth
	
	// Solve Type:
	//  * Depths W/L/D
	//  * W/L/D
	//  * Depths W
	//  * W
	
	std::vector<int8_t> result1;
	std::vector<int8_t> result2;
	if ((0)) // backward approaches (depth)
	{
		// expandAll, winOnly, immediateWin
		// These four (expand all) are far too slow
		BackwardSolver<true, true, true> backward1;
		BackwardSolver<true, true, false> backward2;
		BackwardSolver<true, false, true> backward3;
		BackwardSolver<true, false, false> backward4;
		BackwardSolver<false, true, true> backward5;
		BackwardSolver<false, true, false> backward6;
		BackwardSolver<false, false, true> backward7;
		BackwardSolver<false, false, false> backward8;
		backward1.SolveGame(result1);
		backward2.SolveGame(result1);
		backward3.SolveGame(result1);
		backward4.SolveGame(result1);
		backward5.SolveGame(result1);
		backward6.SolveGame(result1);
		backward7.SolveGame(result1);
		backward8.SolveGame(result1);
	}
	if ((0)) // forward approaches
	{
		// strictDepth, winOnly, repairDepth, backProp
		ForwardSolver<true, false, false> forward1;
		ForwardSolver<true, true, false> forward2;
		ForwardSolver<false, false, false> forward3;
		ForwardSolver<false, true, false> forward4;
		forward1.SolveGame(result2);
		forward2.SolveGame(result2);
		forward3.SolveGame(result2);
		forward4.SolveGame(result2);
	}
	if ((0)) // hybrid approaches
	{
		// strictDepth, winOnly, repairDepth, backProp
		// These two don't really make sense
		//		ForwardSolver<true, false, false, true> forward1;
		//		ForwardSolver<true, true, false, true> forward2;
		ForwardSolver<false, false, true> forward3;
		ForwardSolver<false, true, true> forward4;
		//		forward1.SolveGame(result2);
		//		forward2.SolveGame(result2);
		forward3.SolveGame(result2);
		forward4.SolveGame(result2);
	}
	if ((0)) // parallelism comparison
	{
		BackwardSolver<false, true, true> backward5;
		BackwardSolver<false, true, false> backward6;
		BackwardSolver<false, false, true> backward7;
		BackwardSolver<false, false, false> backward8;
		backward5.SolveGame(result1, true); // option 0
		backward5.SolveGame(result1, false);
		backward6.SolveGame(result1, true); // option 1
		backward6.SolveGame(result1, false);
		backward7.SolveGame(result1, true);
		backward7.SolveGame(result1, false);
		backward8.SolveGame(result1, true);
		backward8.SolveGame(result1, false);
		
		ForwardSolver<true, false, false> forward1;
		ForwardSolver<true, true, false> forward2;
		ForwardSolver<false, false, false> forward3;
		ForwardSolver<false, true, false> forward4;
		forward1.SolveGame(result2, true);
		forward1.SolveGame(result2, false);
		forward2.SolveGame(result2, true);
		forward2.SolveGame(result2, false);
		forward3.SolveGame(result2, true);
		forward3.SolveGame(result2, false);
		forward4.SolveGame(result2, true);
		forward4.SolveGame(result2, false);
		
		ForwardSolver<false, false, true> forward3f;
		ForwardSolver<false, true, true> forward4f;
		forward3f.SolveGame(result2, true); // option 2
		forward3f.SolveGame(result2, false);
		forward4f.SolveGame(result2, true);
		forward4f.SolveGame(result2, false);
	}
	
	if ((0)) // for bigger board
	{
		// expandAll, winOnly, immediateWin
		//		BackwardSolver<false, true, true> backward5;
		//		BackwardSolver<false, true, false> backward6;
		BackwardSolver<true, false, true> backward7;
		//		backward5.SolveGame(result1, false); // option 0
		//		backward5.SolveGame(result1, true); // option 0
		//		backward6.SolveGame(result1, true); // option 1
		backward7.SolveGame(result1, true); // option 1
		
		//		ForwardSolver<false, false, true> forward3f;
		//		forward3f.SolveGame(result2, true); // option 2
		
	}
}
