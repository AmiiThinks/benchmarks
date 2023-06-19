//
//  test_proofTree.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 8/12/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include "CCheckers.h"
#include "FileUtils.h"
#include <vector>
#include "SimpleSolver.h"
#include "PrioritySolver.h"
#include <assert.h>
#include <algorithm>
#include <deque>
#include "CCSeparation.h"

std::vector<bool> data;
std::vector<uint8_t> saData;
std::vector<bool> inTree;
void BuildMinNode(CCheckers &cc, CCState &s, int depth, CCMove *lastMax, CCMove *lastMin);
void BuildMaxNode(CCheckers &cc, CCState &s, int depth, CCMove *lastMax, CCMove *lastMin);
uint64_t nodesVisited = 0;
CCMove *BestMaxMove(CCheckers &cc, CCState &s);
CCMove *BestMinMove(CCheckers &cc, CCState &s);
void BFS();
CCState far;
void ShowNodeCounts();
std::vector<bool> legalSpots;

int main(int argc, char * const argv[])
{
	CCheckers cc;
	CCState s;
	cc.Reset(s);
	s.Reverse();

//	BuildSASDistance(saData);
//	printf("%d across the board\n", saData[cc.rankPlayer(s, 0)]);
//	return 0;
	
#ifdef BOARD_49_2
	legalSpots.resize(NUM_SPOTS);
	legalSpots[0] = true;
	legalSpots[1] = legalSpots[3] = legalSpots[6] = legalSpots[10] = legalSpots[15] = true;
//	legalSpots[1] = legalSpots[2] = true;
//	legalSpots[3] = legalSpots[4] = true;
//	legalSpots[6] = legalSpots[7] = true;
//	legalSpots[10] = legalSpots[11] = true;
//	legalSpots[15] = legalSpots[16] = true;
	legalSpots[21] = legalSpots[22] = true;
	legalSpots[28] = legalSpots[29] = true;
	legalSpots[34] = legalSpots[35] = true;
	legalSpots[39] = legalSpots[40] = true;
	legalSpots[43] = legalSpots[44] = true;
	legalSpots[47] = legalSpots[46] = true;
	legalSpots[48] = true;
#endif
	
	// 0. compute all distances to goal states
//	SimpleSolveGame(data);
//
//	ShowNodeCounts();
//	return 0;
	
#ifdef BOARD_49_3
	ReadData(data, "3-piece-all");
#else
	PrioritySolver(data, 0, false);
#endif
	
	inTree.resize(data.size());
	
	// 1. load data file
	//ReadData(data, "2-piece-proof");

	clrscr();
	
	// 2. start at root and build a proof tree

	printf("Building sas distances\n"); fflush(stdout);
	BuildSASDistance(saData);
//	// 3. Build table of most winning states
	saData.resize(cc.getMaxSinglePlayerRank());
	for (unsigned int x = 0; x < data.size(); x++)
	{
		if (data[x] == 0)
			continue;
		cc.unrank(x, s);
		uint64_t pRank = cc.rankPlayer(s, 0);
		saData[pRank]++;
	}

	
//	clrscr();
	BuildMaxNode(cc, s, 0, 0, 0);
//	gotoxy(1, 37);

	//BFS();
	//far.Print();

//	uint64_t samax = cc.getMaxSinglePlayerRank();
//	std::vector<uint16_t> saranks(samax);
//	for (uint64_t x = 0; x < cc.getMaxRank(); x++)
//	{
//		if (inTree[x])
//		{
//			cc.unrank(x, s);
//			uint64_t sarank = cc.rankPlayer(s, 0);
//			s.Print();
//			printf("(State %llu)\n", x);
//			saranks[sarank]++;
//		}
//	}
//	int cnt = 0;
//	printf("Player 0 states\n");
//	for (uint64_t x = 0; x < samax; x++)
//	{
//		if (saranks[x])
//		{
//			cc.unrankPlayer(x, s, 0);
//			s.Print();
//			printf("%d occurrences\n", saranks[x]);
//			cnt++;
//		}
//	}
//	clrscr();
//	for (uint64_t x = 0; x < samax; x++)
//	{
//		if (saranks[x])
//		{
//			cc.unrankPlayer(x, s, 0);
//			printf("%d occurrences\n", saranks[x]);
//		}
//	}
//	printf("%d unique states for player 0\n", cnt);
#ifdef BOARD_49_3
	WriteData(inTree, "minProofTree-3");
	WriteData(data, "minProofTree-3before");
#endif

#ifdef BOARD_49_2
	WriteData(inTree, "minProofTree-2");
	WriteData(data, "minProofTree-2before");
#endif
	
#ifdef BOARD_49_1
	WriteData(inTree, "minProofTree-1");
	WriteData(data, "minProofTree-1before");
#endif

#ifdef BOARD_9_1
	WriteData(inTree, "minProofTree-9-1");
	WriteData(data, "minProofTree-9-1before");
#endif
	printf("%llu nodes in proof tree; %lu nodes in game\n", nodesVisited, data.size());
}

void BuildMaxNode(CCheckers &cc, CCState &s, int depth, CCMove *lastMax, CCMove *lastMin)
{
	fflush(stdout);
	uint64_t initRank = cc.rank(s);
	if (inTree[initRank] == true)
		return;
	inTree[initRank] = true;
	assert(data[initRank]);
	nodesVisited++;
	
	if (cc.Done(s))
		return;
//
//	s.Print();
//	printf("At depth %d; max to move\n", depth);
//	inTree[cc.rank(s)] = true;
	CCMove *best = BestMaxMove(cc, s);

	if (depth < 150)
	{
		gotoxy(5+20*((int)depth/30), 1+(depth)%30);
		printf("%3d max: 1 child\n", depth);
	}
	else { gotoxy(5, 35); printf("%5d\n", depth); }
	//std::cout << "Taking move: " << *best << std::endl;
	cc.ApplyMove(s, best);
	BuildMinNode(cc, s, depth+1, best, lastMin);
	cc.UndoMove(s, best);
	cc.freeMove(best);
}

void BuildMinNode(CCheckers &cc, CCState &s, int depth, CCMove *lastMax, CCMove *lastMin)
{
	uint64_t initRank = cc.rank(s);

	if (inTree[initRank] == true)
		return;
	inTree[initRank] = true;
	assert(cc.Winner(s) == 0 || data[initRank]);
	nodesVisited++;

	if (cc.Done(s))
		return;

	CCMove *m = cc.getMoves(s);
	
//	s.Print();
//	printf("At depth %d; min to move\n", depth);

	// choose best min move first to force an interesting proof tree
	CCMove *tmp = BestMinMove(cc, s);
	tmp->next = m;
	m = tmp;
	
	for (CCMove *t = m; t; t = t->next)
	{
		if (depth < 150)
		{
			gotoxy(5+20*((int)depth/30), 1+(depth)%30);
			printf("%3d min: %3d left\n", depth, t->length());
		}
		else { gotoxy(5, 35); printf("%5d\n", depth); }
		
		// don't reverse the last move we took
		if (lastMin && (lastMin->from == t->to && lastMin->to == t->from))
			continue;
		
		cc.ApplyMove(s, t);
		BuildMaxNode(cc, s, depth+1, lastMax, t);
		cc.UndoMove(s, t);
	}
	cc.freeMove(m);
}

CCMove *BestMaxMove(CCheckers &cc, CCState &s)
{
	assert(s.toMove == 0);
	assert(data[cc.rank(s)]);
	int bestCnt = 1000;
	uint8_t cnt = 0;
	CCMove *m = cc.getMoves(s);
	CCMove *best = 0;
	// taking winning moves immediately increases the size of the proof tree!
//	for (CCMove *t = m; t; t = t->next)
//	{
//		cc.ApplyMove(s, t);
//		if (cc.Winner(s) == 0)
//		{
//			cc.UndoMove(s, t);
//			best = t->clone(cc);
//			cc.freeMove(m);
//			return best;
//		}
//		cc.UndoMove(s, t);
//	}
	for (CCMove *t = m; t; t = t->next)
	{
#ifdef BOARD_49_2
		if (!legalSpots[t->to])
			continue;
#endif
		cc.ApplyMove(s, t);
		if (inTree[cc.rank(s)])
		{
			cc.UndoMove(s, t);
//			continue;
			best = t->clone(cc);
			cc.freeMove(m);
			return best;
		}
//		CCMove *mm = cc.getMoves(s);
//
//		// cnt is the distance to the goal
//		cnt = 0;
//		for (CCMove *tt = mm; tt; tt = tt->next)
//		{
//			cc.ApplyMove(s, tt);
//
//			//CCState tmp(s);
//			//tmp.Reverse();
//			
//			//cnt = std::max(cnt, saData[cc.rankPlayer(tmp, 0)]);
//			if (data[cc.rank(s)])
//			{
//				cnt++;
//			}
//			//cnt += tt->to-tt->from;
//			
//			cc.UndoMove(s, tt);
//		}
//		cc.freeMove(mm);

		// distance to goal
		CCState tmp(s);
		tmp.Reverse();
		cnt = saData[cc.rankPlayer(tmp, 0)];

//		std::cout << "Move " << *t << " puts us " << (int)cnt << " away" << std::endl;
		if (data[cc.rank(s)] && ((cnt < bestCnt) || (best == 0)))
		{
			bestCnt = cnt;
			cc.freeMove(best);
			best = t->clone(cc);
		}
		cc.UndoMove(s, t);
	}
	if (best == 0)
	{
		s.Print();
		//best = m->clone(cc);
		assert(!"No winning move!");
	}
	cc.freeMove(m);
	cc.ApplyMove(s, best);
	cc.UndoMove(s, best);
	//	std::cout << "Taking " << *best << std::endl;
	return best;
}

CCMove *BestMinMove(CCheckers &cc, CCState &s)
{
	assert(s.toMove == 1);
	int bestCnt = 1000;
	uint8_t cnt = 0;
	CCMove *m = cc.getMoves(s);
	CCMove *best = 0;
	// TODO: Is this code even relevatn here???
//	for (CCMove *t = m; t; t = t->next)
//	{
//		cc.ApplyMove(s, t);
//		if (cc.Done(s))
//		{
//			cc.UndoMove(s, t);
//			best = t->clone(cc);
//			cc.freeMove(m);
//			return best;
//		}
//		cc.UndoMove(s, t);
//	}

	for (CCMove *t = m; t; t = t->next)
	{
		cc.ApplyMove(s, t);
		cnt = saData[cc.rankPlayer(s, 1)];

		if (data[cc.rank(s)] && ((cnt < bestCnt) || (best == 0)))
		{
			bestCnt = cnt;
			cc.freeMove(best);
			best = t->clone(cc);
		}
		cc.UndoMove(s, t);
	}
	cc.freeMove(m);
	//	std::cout << "Taking " << *best << std::endl;
	return best;
}

//std::vector<bool> data;
//std::vector<uint8_t> saData;
//std::vector<bool> inTree;

void BFS()
{
	CCheckers cc;
	CCState s;
	
	cc.Reset(s);

	std::vector<CCState> maxPlayer;
	std::vector<CCState> minPlayer;

	maxPlayer.push_back(s);
	
	while (maxPlayer.size() > 0 || minPlayer.size() > 0)
	{
		printf("%ld in max queue\n", maxPlayer.size());
		while (maxPlayer.size() > 0)
		{
			CCState next = maxPlayer.back();
			far = next;
			maxPlayer.pop_back();

			CCMove *m = BestMaxMove(cc, next);
			cc.ApplyMove(next, m);
			
			uint64_t rank = cc.rank(next);

			if (!inTree[rank])
			{
				inTree[rank] = true;
				nodesVisited++;
				if (cc.Done(next) || TestSeparationWin(cc, next, saData, 0) == kProvenWin)
				{
					// ignore; we're done
				}
				else {
					minPlayer.push_back(next);
				}
			}
			cc.freeMove(m);
		}

		printf("%ld in min queue\n", minPlayer.size());
		while (minPlayer.size() > 0)
		{
			CCState next = minPlayer.back();
			far = next;
			minPlayer.pop_back();
			
			CCMove *m = cc.getMoves(next);
			for (CCMove *t = m; t; t = t->next)
			{
				cc.ApplyMove(next, t);
				uint64_t rank = cc.rank(next);
				if (!inTree[rank])
				{
					inTree[rank] = true;
					nodesVisited++;
					if (cc.Done(next) || TestSeparationWin(cc, next, saData, 0) == kProvenWin)
					{
						// ignore; we're done
					}
					else {
						maxPlayer.push_back(next);
					}
				}
				cc.UndoMove(next, t);
			}
			cc.freeMove(m);
		}
	}
}


void ShowNodeCounts()
{
	CCheckers cc;
	CCState s;
	
	cc.Reset(s);
	inTree.resize(cc.getMaxRank());
	std::vector<uint64_t> q1, q2;
	std::vector<double> counts;
	
	q1.push_back(cc.rank(s));
	
	while (q1.size() > 0 || q2.size() > 0)
	{
		printf("%ld in queue", q1.size());
		counts.push_back(q1.size());
		if (counts.size() > 1)
			printf(" BF: %1.2f\n", counts[counts.size()-1]/counts[counts.size()-2]);
		else
			printf("\n");
		while (q1.size() > 0)
		{
			CCState next;
			cc.unrank(q1.back(), next);
			far = next;
			q1.pop_back();
			
			CCMove *m = cc.getMoves(next);
			for (CCMove *t = m; t; t = t->next)
			{
				cc.ApplyMove(next, t);
				uint64_t rank = cc.rank(next);
				if (!inTree[rank])
				{
					inTree[rank] = true;
					nodesVisited++;
					if (cc.Done(next))
					{
						// ignore; we're done
					}
					else {
						q2.push_back(rank);
					}
				}
				cc.UndoMove(next, t);
			}
			cc.freeMove(m);
		}

		printf("%ld in queue", q2.size());
		if (counts.size() > 1)
			printf(" BF: %1.2f\n", counts[counts.size()-1]/counts[counts.size()-2]);
		else
			printf("\n");
		counts.push_back(q2.size());
		while (q2.size() > 0)
		{
			CCState next;
			cc.unrank(q2.back(), next);
			
			far = next;
			q2.pop_back();
			
			CCMove *m = cc.getMoves(next);
			for (CCMove *t = m; t; t = t->next)
			{
				cc.ApplyMove(next, t);
				uint64_t rank = cc.rank(next);
				if (!inTree[rank])
				{
					inTree[rank] = true;
					nodesVisited++;
					if (cc.Done(next))
					{
						// ignore; we're done
					}
					else {
						q1.push_back(rank);
					}
				}
				cc.UndoMove(next, t);
			}
			cc.freeMove(m);
		}
	}
}

