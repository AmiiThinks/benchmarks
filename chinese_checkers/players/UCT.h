//
//  UCT.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/28/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef __CC_UCT__UCT__
#define __CC_UCT__UCT__

#include <stdio.h>
#include <string>
#include <limits>
#include <math.h>
#include "CCUtils.h"
#include "Player.h"
#include "CCheckers.h"
#include "Timer.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class RandomPlayout {
public:
	const char *GetName() { return "rand"; }
	CCMove *GetNextAction(CCheckers *cc, CCState &s)
	{
		CCMove *m = cc->getMovesForward(s);
		if (m == 0)
			m = cc->getMoves(s);
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
};

class BestPlayout {
public:
	const char *GetName() { return "best"; }
	CCMove *GetNextAction(CCheckers *cc, CCState &s)
	{
		if ((random()%100) <= 5)
			return p.GetNextAction(cc, s);
		CCMove *m = cc->getMovesForward(s);
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
		return m;
	}
private:
	RandomPlayout p;
};

class BackPlayout {
public:
	const char *GetName() { return "back"; }
	CCMove *GetNextAction(CCheckers *cc, CCState &s)
	{
		if ((random()%100) <= 5)
			return p.GetNextAction(cc, s);
		int which;
		if (s.toMove == 0)
			which = NUM_PIECES-1;
		else
			which = 0;
		CCMove *m = cc->getMovesForPiece(s, which);
		if (m == 0)
			return p.GetNextAction(cc, s);

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
		return m;
	}
private:
	RandomPlayout p;
};


struct UCTNode {
	uint32_t numChildren;
	uint32_t firstChild;
	uint32_t numSamples;
	uint32_t toMove;
	CCMove m;
	int from;
	int to;
	double totalPayoff;
};


class Node {
public:
    Node() {from = 0; to = 0; numSamples = 0;}
    int from;
    int to;
    int numSamples;
};


template <class playoutModule, class evalModule>
class UCT : public Player {
public:
	UCT(double C, playoutModule *m, evalModule *e)
	:C(C), m(m), e(e) { playOutDepth = 10; treeExpansionLimit = 0; initFromDB = false; }
	CCMove *GetNextAction(CCheckers *cc, const CCState &s,
						  double &value, double timeLimit, int depthLimit = 100000);
	virtual const char *GetName();
	int playOutDepth;
	int treeExpansionLimit;
	void Reset() {}
    std::vector<Node> stats;
private:
	CCMove *PickBestMove(CCheckers *cc, CCState s);
	double SelectLeaf(uint32_t node);
	uint32_t SelectBestChild(uint32_t node);
	bool IsLeaf(uint32_t node);
	void Expand(uint32_t node);
	void InitFromDB(uint32_t node, CCState &s);
	double DoPlayout(uint32_t node);
	double GetUCBVal(uint32_t node, uint32_t parent);
	std::string name;
	double C;
	bool initFromDB;
	std::vector<UCTNode> tree;
	playoutModule *m;
	evalModule *e;
	CCheckers *cc;
	CCState currState;
	uint64_t playoutLength, samples;
};


template <class playoutModule, class evalModule>
CCMove *UCT<playoutModule, evalModule>::PickBestMove(CCheckers *cc, CCState s)
{
	int who = s.toMove;
	CCMove *m = cc->getMovesForward(s);
	double bestVal = std::numeric_limits<double>::lowest();
	CCMove *best = 0;
	for (CCMove *t = m; t; t = t->next)
	{
		cc->ApplyMove(s, t);
		double tmpVal = e->eval(cc, s, who);
		if (tmpVal > bestVal)
		{
			bestVal = tmpVal;
			best = t;
		}
		std::cerr << *t << " has value " << tmpVal << "\n";
		cc->UndoMove(s, t);
	}
	best = best->clone(*cc);
	cc->freeMove(m);
	std::cerr << GetName() << " taking action " << *best << "\n";
	return best;
}


template <class playoutModule, class evalModule>
CCMove *UCT<playoutModule, evalModule>::GetNextAction(CCheckers *cc, const CCState &s,
					  double &value, double timeLimit, int)
{
	Timer t;
	t.StartTimer();

	if (e->perfectEval(s) && e->canEval(s))
	{
		return PickBestMove(cc, s);
	}

	
	samples = 0;
	playoutLength = 0;
	this->cc = cc;
	stats.clear();
	stats.resize(1);
	tree.clear();
	tree.resize(1);
	tree[0].toMove = s.toMove;
	this->currState = s;
	Expand(0);
	if (initFromDB)
		InitFromDB(0, currState);
	
	while (t.GetElapsedTime() < timeLimit)
	{
		samples++;
		this->currState = s;
		SelectLeaf(0);
	}
	UCTNode &root = tree[0];
	Node tmp_node;
	int best = root.firstChild;
	for (int x = root.firstChild; x < root.firstChild + root.numChildren; x++)
	{
	    
	    tmp_node.from = tree[x].m.from;
	    tmp_node.to = tree[x].m.to;
	    tmp_node.numSamples = tree[x].numSamples;
	    stats.push_back(tmp_node);
	    
		std::cerr << tree[x].m << " " << tree[x].numSamples << " samples; avg: ";
		std::cerr << tree[x].totalPayoff / tree[x].numSamples << "\n";
		if (tree[x].totalPayoff / tree[x].numSamples >
			tree[best].totalPayoff / tree[best].numSamples)
		{
			best = x;
		}
	}
	value = tree[best].totalPayoff / tree[best].numSamples;
	std::cerr << GetName() << " taking action " << tree[best].m << "\n";
	std::cerr << "Tree size: " << tree.size() << " Avg playout length: " << playoutLength / samples << "\n";
	return tree[best].m.clone(*cc);
}

template <class playoutModule, class evalModule>
const char *UCT<playoutModule, evalModule>::GetName()
{
	name = "UCT-";
	name += "C"+std::to_string(int(C))+"-";
	name += "pd"+std::to_string(playOutDepth)+"-";
	name += "el"+std::to_string(treeExpansionLimit)+"-";
	if (initFromDB)
		name += "initdb-";
	name += m->GetName();
	name += "-";
	name += e->GetName();
	return name.c_str();
}

template <class playoutModule, class evalModule>
double UCT<playoutModule, evalModule>::SelectLeaf(uint32_t node)
{
	if (node != 0)
		cc->ApplyMove(currState, &tree[node].m);
	
	double payoff;
	tree[node].numSamples++;
	if (IsLeaf(node))
	{
		if (tree[node].numSamples > treeExpansionLimit)
		{
			Expand(node);
		}
		payoff = DoPlayout(node);
	}
	else {
		payoff = SelectLeaf(SelectBestChild(node));
	}
	if (tree[node].toMove == tree[0].toMove)
	{
		tree[node].totalPayoff += payoff;
	}
	else {
		tree[node].totalPayoff -= payoff;
	}
	return payoff;
}

template <class playoutModule, class evalModule>
uint32_t UCT<playoutModule, evalModule>::SelectBestChild(uint32_t node)
{
	UCTNode &n = tree[node];
	int best = n.firstChild;
	double bestVal = std::numeric_limits<double>::lowest();
	for (int x = n.firstChild; x < n.firstChild + n.numChildren; x++)
	{
		if (tree[x].numSamples == 0)
			return x;
		double val = GetUCBVal(x, node);
		if (val > bestVal)
		{
			best = x;
			bestVal = val;
		}
	}
	return best;
}

template <class playoutModule, class evalModule>
double UCT<playoutModule, evalModule>::GetUCBVal(uint32_t node, uint32_t parent)
{
	double val = tree[node].totalPayoff/tree[node].numSamples;
	val += C*sqrt(log(tree[parent].numSamples)/tree[node].numSamples);
	return val;
}


template <class playoutModule, class evalModule>
double UCT<playoutModule, evalModule>::DoPlayout(uint32_t node)
{
	int depth = 0;
	while (true)
	{
		playoutLength++;
		if (cc->Done(currState))
		{
			return e->eval(cc, currState, tree[0].toMove);
		}
		if ((depth > playOutDepth || e->perfectEval(currState)) && e->canEval(currState))
		{
			// TODO: add 
			return e->eval(cc, currState, tree[0].toMove);
		}
		CCMove *move = m->GetNextAction(cc, currState);
		cc->ApplyMove(currState, move);
		cc->freeMove(move);
		depth++;
	}
}

template <class playoutModule, class evalModule>
bool UCT<playoutModule, evalModule>::IsLeaf(uint32_t node)
{
	return cc->Done(currState) || tree[node].numChildren == 0;
}


template <class playoutModule, class evalModule>
void UCT<playoutModule, evalModule>::Expand(uint32_t node)
{
	if (cc->Done(currState))
		return;
	
	CCMove *m;
//	if (node == 0)
//		m = cc->getMoves(currState);
//	else
	m = cc->getMovesForward(currState);
	tree[node].firstChild = static_cast<uint32_t>(tree.size());
	tree[node].numChildren = m->length();
	tree.resize(tree.size()+m->length());
	uint32_t curr = tree[node].firstChild;
	for (CCMove *t = m; t; t = t->next)
	{
		tree[curr].numChildren = 0;
		tree[curr].firstChild = -1;
		tree[curr].numSamples = 0;
		tree[curr].toMove = currState.toMove;
		tree[curr].m = *t;
		tree[curr].m.next = 0;
		tree[curr].totalPayoff = 0;
		curr++;
	}
	cc->freeMove(m);
}

template <class playoutModule, class evalModule>
void UCT<playoutModule, evalModule>::InitFromDB(uint32_t node, CCState &s)
{
	const int count = 20;
	int dthem = GetDepth("/data/cc/rev/", s, 1-s.toMove);
	uint32_t first = tree[node].firstChild;
	for (uint32_t c = 0; c < tree[node].numChildren; c++)
	{
		cc->ApplyMove(s, &tree[first+c].m);
		int val = GetDepth("/data/cc/rev/", s, 1-s.toMove);
		cc->UndoMove(s, &tree[first+c].m);
		tree[first+c].totalPayoff += (dthem-val)*count;
		tree[first+c].numSamples += count;
	}
}


#endif /* defined(__CC_UCT__UCT__) */
