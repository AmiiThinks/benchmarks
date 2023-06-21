//
//  PrioritySolver.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 7/23/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include <vector>
#include <deque>
#include <stdio.h>
#include <algorithm>
#include "PrioritySolver.h"
#include "CCheckers.h"
#include "FileUtils.h"
#include "CCSeparation.h"

std::vector<uint8_t> startDist;
std::vector<uint8_t> pathDist;
std::vector<uint8_t> boardWeight;

inline int costFunction(uint64_t state);
inline int costFunction(CCheckers &cc, CCState &s);

class priorityData {
public:
	priorityData(uint64_t s, uint32_t p)
	:state(s), priority(p) {}
	uint64_t state;
	uint32_t priority;
};

struct priorityCompare {
	bool operator()(const priorityData &i1, const priorityData &i2) const
	{
		return (i1.priority > i2.priority);
	}
};


void AddToQueue(std::vector<priorityData> &queue, uint64_t state, uint32_t priority)
{
	priorityCompare p;
	queue.push_back(priorityData(state, priority));
	std::push_heap(queue.begin(), queue.end(), p);
}

uint64_t RemoveFromQueue(std::vector<priorityData> &queue)
{
	priorityCompare p;
	uint64_t data;
	std::pop_heap(queue.begin(), queue.end(), p);
	data = queue.back().state;
	queue.pop_back();
	return data;
}

// return estimated cost from this state to the start
inline int costFunction(uint64_t state)
{
	//return (3*pathDist[state]+startDist[state]);
	CCheckers cc;
	CCState s;
	cc.unrankPlayer(state, s, 0);
	return costFunction(cc, s);
}

inline int costFunction(CCheckers &cc, CCState &s)
{
	uint64_t state = cc.rankPlayer(s, 0);
	CCState t = s;
	t.Reverse();
	uint64_t oppState = cc.rankPlayer(t, 1);
	//return costFunction(cc.rankPlayer(s, 0));

	int cost = (5*pathDist[state]+startDist[state]);

	for (int y = 0; y < NUM_PIECES; y++)
		cost += 3*boardWeight[s.pieces[0][y]];

	return 3*cost+2*startDist[oppState]-pathDist[oppState];
	//-100*pathDist[cc.rankPlayer(s, 0)]-startDist[cc.rankPlayer(s, 0)];
//	return 0;
//	int maxr=-1, minr=100;
//	for (int x = 0; x < NUM_PIECES; x++)
//	{
//		int tmpx, tmpy;
//		cc.toxy(s.pieces[0][x], tmpx, tmpy);
//		if (tmpy > maxr)
//			maxr = tmpy;
//		cc.toxy(s.pieces[1][x], tmpx, tmpy);
//		if (tmpy < minr)
//			minr = tmpy;
//	}
//	return (minr - maxr);

	// 4816507
	//return random();
	
	// 4747186
	//return abs(cc.startDistance(s, 0)-cc.startDistance(s, 1));

	// 4351852
	//return -abs(cc.startDistance(s, 0)-cc.startDistance(s, 1));

	// 4351852
	//return -abs(cc.startDistance(s, 0)-cc.startDistance(s, 1))-abs(cc.goalDistance(s, 0)-cc.goalDistance(s, 1));

	// 3834913
	//return -cc.startDistance(s, 0);

	//return 0;
	//return -d[cc.rankPlayer(s, 0)];
	//return (d[cc.rankPlayer(s, 0)]+d[cc.rankPlayer(s, 1)]);
	
	// 4373856
	//return abs(2*cc.goalDistance(s, 0)-cc.goalDistance(s, 1));

	// 4548137
	//return -abs(cc.goalDistance(s, 0)-2*cc.goalDistance(s, 1));

	// 4756771
	//return abs(cc.goalDistance(s, 0)-2*cc.goalDistance(s, 1));
}

void BuildBoardWeight(std::vector<uint64_t> &path)
{
	CCheckers cc;
	CCState s;
	boardWeight.resize(NUM_SPOTS);

	for (int x = 0; x < NUM_SPOTS; x++)
		boardWeight[x] = 3;
	for (unsigned int x = 0; x < path.size(); x++)
	{
		cc.unrankPlayer(path[x], s, 0);
		for (int y = 0; y < NUM_PIECES; y++)
			boardWeight[s.pieces[0][y]] = 0;
	}
	for (unsigned int x = 0; x < path.size(); x++)
	{
		cc.unrankPlayer(path[x], s, 0);
		CCMove *m = cc.getMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			if (boardWeight[t->to] == 3)
				boardWeight[t->to] = 1;
		}
		cc.freeMove(m);
	}
}

void ExtractPath(std::vector<uint64_t> &path)
{
	CCheckers cc;
	CCState s;
	cc.Reset(s);
	uint64_t goal = cc.rankPlayer(s, 1);
	uint64_t startRank = cc.rankPlayer(s, 0);
	printf("Starting at goal dist %d\n", startDist[goal]);
	
	while (goal != startRank)
	{
		path.push_back(goal);
		
		cc.unrankPlayer(goal, s, 0);
		CCMove *m = cc.getMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t);
			uint64_t child = cc.rankPlayer(s, 0);
			if (startDist[child] < startDist[goal])
			{
				goal = child;
				printf("Moving to child dist %d\n", startDist[goal]);
				cc.UndoMove(s, t);
				break;
			}
			cc.UndoMove(s, t);
		}
		cc.freeMove(m);
	}
}

void BuildSASDistance(std::vector<uint8_t> &d)
{
	CCheckers cc;
	CCState s;
	std::vector<uint64_t> starts;
	
	cc.Reset(s);
	uint64_t startRank = cc.rankPlayer(s, 0);
	starts.push_back(startRank);

	BuildSASDistance(d, starts);
}

void BuildSASDistance(std::vector<uint8_t> &d, std::vector<uint64_t> &startStates)
{
	CCheckers cc;
	CCState s;
	
	std::deque<uint64_t> q;
	uint64_t maxRank = cc.getMaxSinglePlayerRank();
	d.resize(maxRank);
	printf("%llu entries in SA distance\n", maxRank);
	for (uint64_t val = 0; val < maxRank; val++)
	{
		d[val] = 255;
	}
	cc.Reset(s);

	for (unsigned int x = 0; x < startStates.size(); x++)
	{
		q.push_back(startStates[x]);
		d[startStates[x]] = 0;
	}

	uint64_t next;
	while (q.size() > 0)
	{
		static int cnt = 1;
		if ((++cnt)%10000 == 0)
		{
			printf("%lu     \r", q.size());
			fflush(stdout);
		}
		next = q.front();
		q.pop_front();
		
		cc.unrankPlayer(next, s, 0);
		CCMove *m = cc.getMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.ApplyMove(s, t);
			uint64_t child = cc.rankPlayer(s, 0);
			if (d[child] == 255)
			{
				d[child] = d[next]+1;
				q.push_back(child);
			}
			cc.UndoMove(s, t);
		}
		cc.freeMove(m);
	}
}

void AnalyzeProof(std::vector<bool> &wins)
{
	return;
	
	std::vector<int> values;
	std::vector<int> counts;
	int total = 0;

	CCheckers cc;
	CCState s;

	for (unsigned int x = 0; x < wins.size(); x++)
	{
		if (wins[x])
		{
			cc.unrank(x, s);
			uint64_t sa = cc.rankPlayer(s, 0);
			
			total++;
			bool found = false;
			for (unsigned int y = 0; y < values.size(); y++)
			{
				if (costFunction(sa) == values[y])
				{
					counts[y]++;
					found = true;

//					if (costFunction(sa) == 0)
//					{
//						CCheckers cc;
//						CCState s;
//						cc.unrank(x, s);
//						s.Print();
//						//printf("SA: %llu Path: %llu\n", pathDist[x], startDist[x]);
//					}

					break;
				}
			}
			if (!found)
			{
				values.push_back(costFunction(sa));
				counts.push_back(1);
//				if (costFunction(x) == 0)
//				{
//					CCheckers cc;
//					CCState s;
//					cc.unrank(x, s);
//					s.Print();
//				}
			}
		}
	}
//	for (int x = 0; x < values.size(); x++)
//	{
//		for (unsigned int y = 0; y < values.size()-x; y++)
//		{
//			if (values[y] > values[y+1])
//			{
//				int tmp = values[y];
//				values[y] = values[y+1];
//				values[y+1] = tmp;
//
//				tmp = counts[y];
//				counts[y] = counts[y+1];
//				counts[y+1] = tmp;
//			}
//		}
//	}
	for (unsigned int y = 0; y < values.size(); y++)
	{
		//if (counts[y] > 10)
		printf("%d: %d\n", values[y], counts[y]);
	}
	printf("%d out of %llu entries used\n", total, (uint64_t)wins.size());
}


void PrioritySolver(std::vector<bool> &wins, const char *outputFile, bool stopAfterProof)
{
	int provingPlayer = 0;
	CCheckers cc;
	CCState s;

	std::vector<priorityData> winStates;
	std::vector<uint64_t> path;
	BuildSASDistance(startDist);
	printf("Done building SA distances\n");
	ExtractPath(path);
	printf("%d states on optimal SA path\n", (int)path.size());
	for (unsigned int x = 0; x < path.size(); x++)
	{
		cc.unrankPlayer(path[x], s, 0);
		s.Print();
	}
	BuildSASDistance(pathDist, path);
	BuildBoardWeight(path);
	//exit(0);

	cc.Reset(s);
	uint64_t root = cc.rank(s);
	uint64_t maxRank = cc.getMaxRank();
	wins.resize(maxRank);
	//std::vector<bool> wins(maxRank);
	uint64_t winningStates = 0;
	uint64_t illegalStates = 0;
	uint64_t legalStates = 0;
	printf("Finding won positions for player %d; %llu total positions\n", provingPlayer, maxRank);
	float perc = 0;

	uint64_t maxPlayerRank = cc.getMaxSinglePlayerRank();
//	uint64_t rankPlayer(CCState &s, int who); // TODO: write these
//	// returns true if it is a valid unranking given existing pieces
//	bool unrankPlayer(uint64_t, CCState &s, int who); // TODO: write these

	// just go through single-agent states
	for (uint64_t val = 0; val < maxPlayerRank; val++)
	{
		if (100.0*val/maxPlayerRank >= perc)
		{
			perc+=5;
			std::cout << val << " of " << maxPlayerRank << " " << 100.0*val/maxPlayerRank << "% complete " << 
			legalStates << " legal " << winningStates << " winning " << illegalStates << " illegal" << std::endl;
		}
		if (cc.unrankPlayer(val, s, 1-provingPlayer))
		{
			// if we can't move our pieces into the goal state, ignore
			if (!cc.MovePlayerToGoal(s, provingPlayer))
				continue;
			legalStates++;
			if (cc.Done(s))
			{
				if (cc.Winner(s) == provingPlayer)
				{
					if (s.toMove == 1-provingPlayer)
					{
						//s.PrintASCII();
						uint64_t gameRank = cc.rank(s);
						wins[gameRank] = true;
						AddToQueue(winStates, gameRank, costFunction(cc, s));
						winningStates++;
					}
					else {
						//printf("###"); s.PrintASCII();
						illegalStates++;
					}
				}
			}
		}
	}

//	for (uint64_t val = 0; val < maxRank; val++)
//	{
//		if (100.0*val/maxRank >= perc)
//		{
//			perc+=5;
//			std::cout << val << " of " << maxRank << " " << 100.0*val/maxRank << "% complete " << 
//			legalStates << " legal " << winningStates << " winning " << illegalStates << " illegal" << std::endl;
//		}
//		if (cc.unrank(val, s))
//		{
//			legalStates++;
//			if (cc.Done(s))
//			{
//				if (cc.Winner(s) == provingPlayer)
//				{
//					if (s.toMove == 1-provingPlayer)
//					{
//						wins[val] = true;
//						AddToQueue(winStates, val, costFunction(cc, s));
//						s.PrintASCII();
//						winningStates++;
//					}
//					else {
//						illegalStates++;
//					}
//				}
//			}
//		}
//		else {
//		}
//	}
	printf("%lld states unranked; %lld were winning; %lld were tech. illegal\n",
		   legalStates, winningStates, illegalStates);

	uint64_t nextBound = 100000;
	while (winStates.size() > 0)
	{
		if (winningStates > nextBound)
		{
			printf("%llu states proven\n", winningStates);
			fflush(stdout);
			nextBound += 100000;
		}
		uint64_t currState = RemoveFromQueue(winStates);
		cc.unrank(currState, s);
				
//		printf("Expanding state cost %d\n", costFunction(cc, s));
//		printf("Working on state %llu\n", currState);
		

//		if (kProvenWin == TestSeparationWin(cc, s, dist, provingPlayer))
//		{
//			printf("Win proven for %d; %llu states in proof\n", provingPlayer, winningStates);
//			s.Print();
//			s.PrintASCII();
//			WriteData(wins, "3-piece-proof");
//			return;
//		}

		// for each unproven successor of s
		CCMove *m = cc.getReverseMoves(s);
		for (CCMove *t = m; t; t = t->next)
		{
			cc.UndoMove(s, t);
			uint64_t succ = cc.rank(s);

			if (cc.Winner(s) == (1-provingPlayer))
			{
				
			}
			// ignore proven states
			else if (wins[succ])
			{
			}
			// need this because of won positions where the goal isn't full of our pieces
			// We can't easily start with them, but we need to add them in places.
			else if ((cc.Winner(s) == provingPlayer) && (s.toMove == 1-provingPlayer))
			{
//				printf("**");
//				s.PrintASCII();
				AddToQueue(winStates, succ, costFunction(cc, s));
			}
			// already know that a successor is proven (proving player)
			else if (s.toMove == provingPlayer)
			{
				wins[succ] = true;
				winningStates++;
				if (succ == root)
				{
					printf("Win proven for %d; %llu states in proof\n", provingPlayer, winningStates);
					if (stopAfterProof)
					{
						if (outputFile != 0)
							WriteData(wins, outputFile);
						AnalyzeProof(wins);
						return;
					}

				}
				AddToQueue(winStates, succ, costFunction(cc, s));
			}
			else { // need all successors to be proven (non-proving player)
				CCMove *nextSucc = cc.getMoves(s);
				bool allChildrenWins = true;
				for (CCMove *tmp = nextSucc; tmp; tmp = tmp->next)
				{
					cc.ApplyMove(s, tmp);
					uint64_t nextChild = cc.rank(s);
					cc.UndoMove(s, tmp);
					if (!wins[nextChild])
					{
						allChildrenWins = false;
						break;
					}
				}
				cc.freeMove(nextSucc);
				if (allChildrenWins)
				{
					wins[succ] = true;
					winningStates++;
					if (succ == root)
					{
						printf("Win proven for %d; %llu states in proof\n", provingPlayer, winningStates);

						if (stopAfterProof)
						{
							if (outputFile != 0)
								WriteData(wins, outputFile);
							AnalyzeProof(wins);
							return;
						}
					}
					AddToQueue(winStates, succ, costFunction(cc, s));
				}
			}
			cc.ApplyMove(s, t);
		}
		cc.freeMove(m);
	}
	printf("Win not proven for %d; %llu states in proof\n", provingPlayer, winningStates);
	if (outputFile != 0)
		WriteData(wins, outputFile);
}

