//
//  UCB.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/20/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include <math.h>
#include "UCB.h"
#include "DistEval.h"
#include "Timer.h"

CCMove *UCB::GetRandomAction(CCheckers *cc, CCState &s)
{
	CCMove *m = cc->getMovesForward(s);
	int count = m->length();
	int which = random()%count;
	while (which != 0)
	{
		which--;
		CCMove *tmp = m;
		m = m->next;
		tmp->next = 0;
		cc->freeMove(tmp);
	}
	cc->freeMove(m->next);
	m->next = 0;
	return m;
}


double UCB::DoRandomPlayout(CCheckers *cc, CCState s)
{
	DistEval e;
	uint64_t count = 0;
	while (!cc->Done(s) && count < 10)
	{
		CCMove *m;
		if ((random()%100) < 20)
		{
			m = GetRandomAction(cc, s);
		}
		else {
			m = cc->getMovesForward(s);
			CCMove *best = m;
			for (CCMove *t = m->next; t; t = t->next)
			{
				int dist1 = cc->distance(t->from,
										cc->getGoal(s.toMove))
				- cc->distance(t->to,
							   cc->getGoal(s.toMove));
				int dist2 = cc->distance(best->from,
										 cc->getGoal(s.toMove))
				- cc->distance(best->to,
							   cc->getGoal(s.toMove));
				if (dist1 > dist2)
					best = t;
			}
			best = best->clone(*cc);
			cc->freeMove(m);
			m = best;
		}
		nodesExpanded++;
		count++;
		cc->ApplyMove(s, m);
		cc->freeMove(m);
	}
	return e.eval(cc, s, rootPlayer)/20.0;
	//std::cerr << count << " nodes in playout\n";
//	if (cc->Winner(s) == rootPlayer)
//		return 1;
//	return 0;
}

double UCB::UCBValue(const UCBData &d, int total)
{
	return d.totalPayoff/d.count + sqrt(2*log(total)/d.count);
}

CCMove *UCB::GetNextAction(CCheckers *cc, const CCState &s, double &bestVal, double timeLimit, int)
{
	std::vector<UCBData> data;
	nodesExpanded = 0;
	Timer t;
	t.StartTimer();
	int total = 0;
	rootPlayer = s.toMove;
	
	CCState state = s;
	CCMove *m = cc->getMovesForward(s);

	for (CCMove *t = m; t; t = t->next)
	{
		cc->ApplyMove(state, t);
		data.push_back({t, 1, DoRandomPlayout(cc, s)});
		total++;
		cc->UndoMove(state, t);
	}
	
	while (t.GetElapsedTime() < timeLimit)
	{
		int best = 0;
		for (int x = 1; x < data.size(); x++)
		{
			if (UCBValue(data[x], total) > UCBValue(data[best], total))
			{
				best = x;
			}
		}

		cc->ApplyMove(state, data[best].m);
		data[best].count++;
		data[best].totalPayoff += DoRandomPlayout(cc, s);
		cc->UndoMove(state, data[best].m);
	}
	
	double best = -1000;
	CCMove *bestMove = 0;
	std::cerr << nodesExpanded << " total nodes expanded\n";
	for (auto &v : data)
	{
		std::cerr << *(v.m) << " has value " << v.totalPayoff/v.count << " ";
		std::cerr << v.count << " samples\n";
		if (v.totalPayoff/v.count > best)
		{
			best = v.totalPayoff/v.count;
			bestMove = v.m;
		}
	}
	bestMove = bestMove->clone(*cc);
	cc->freeMove(data[0].m);
	return bestMove;
}

