////
////  UCTDB.h
////  CC UCT
////
////  Created by Nathan Sturtevant on 5/10/15.
////  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
////
//
//#ifndef CC_UCT_UCTDB_h
//#define CC_UCT_UCTDB_h
//
//#include <stdio.h>
//#include <string>
//#include <limits>
//#include <math.h>
//#include "CCUtils.h"
//#include "Player.h"
//#include "CCheckers.h"
//#include "Timer.h"
//
//struct UCTDBNode {
//	uint32_t numChildren;
//	uint32_t firstChild;
//	uint32_t numSamples;
//	uint16_t toMove;
//	uint8_t p0Dist;
//	uint8_t p1Dist;
//	CCMove m;
//	double totalPayoff;
//};
//
//template <class playoutModule, class evalModule>
//class UCTDB : public Player {
//public:
//	UCTDB(double C, playoutModule *m, evalModule *e)
//	:C(C), m(m), e(e), db(fullDataPrefix, 975) { playOutDepth = 10; treeExpansionLimit = 0; initFromDB = false; firstPiece = 15; }
//	CCMove *GetNextAction(CCheckers *cc, const CCState &s,
//						  double &value, double timeLimit, int depthLimit = 100000);
//	virtual const char *GetName();
//	int playOutDepth;
//	int treeExpansionLimit;
//	void Reset() {}
//private:
//	CCMove *PickBestMove(CCheckers *cc, CCState s);
//	double SelectLeaf(uint32_t node);
//	uint32_t SelectBestChild(uint32_t node);
//	bool IsLeaf(uint32_t node);
//	void Expand(uint32_t node);
//	void InitFromDB(uint32_t node, CCState &s);
//	double DoPlayout(uint32_t node);
//	double GetUCBVal(uint32_t node, uint32_t parent);
//	void RenewDepthEstimates(CCState &currState);
//
//	std::string name;
//	std::vector<UCTDBNode> tree;
//	double C;
//	bool initFromDB;
//	playoutModule *m;
//	evalModule *e;
//	CCheckers *cc;
//	CCState currState;
//	uint64_t playoutLength, samples;
//	
//	int rootDepth0, rootDepth1;
//	bool perfectFromRoot;
//	CCEndGameData db;
//	int firstPiece;
//};
//
//template <class playoutModule, class evalModule>
//void UCTDB<playoutModule, evalModule>::RenewDepthEstimates(CCState &currState)
//{
//	int curr0 = GetDepth(fullDataPrefix, currState, 0);
//	int curr1 = GetDepth(fullDataPrefix, currState, 1);
//	if (curr0 != -1)
//	{
//		while (curr0 < rootDepth0-10)
//			curr0+=15;
//		printf("P0: Was at depth %d, now predicting %d\n", rootDepth0, curr0);
//		rootDepth0 = curr0;
//	}
//	if (curr1 != -1)
//	{
//		while (curr1 < rootDepth1-10)
//			curr1+=15;
//		printf("P1: Was at depth %d, now predicting %d\n", rootDepth1, curr1);
//		rootDepth1 = curr1;
//	}
//}
//
//
//template <class playoutModule, class evalModule>
//CCMove *UCTDB<playoutModule, evalModule>::PickBestMove(CCheckers *cc, CCState s)
//{
//	int who = s.toMove;
//	CCMove *m = cc->getMovesForward(s);
//	double bestVal = std::numeric_limits<double>::lowest();
//	CCMove *best = 0;
//	for (CCMove *t = m; t; t = t->next)
//	{
//		cc->ApplyMove(s, t);
//		double tmpVal = e->eval(cc, s, who);
//		if (tmpVal > bestVal)
//		{
//			bestVal = tmpVal;
//			best = t;
//		}
//		std::cerr << *t << " has value " << tmpVal << "\n";
//		cc->UndoMove(s, t);
//	}
//	best = best->clone(*cc);
//	cc->freeMove(m);
//	std::cerr << GetName() << " taking action " << *best << "\n";
//	return best;
//}
//
//
//template <class playoutModule, class evalModule>
//CCMove *UCTDB<playoutModule, evalModule>::GetNextAction(CCheckers *cc, const CCState &s,
//													  double &value, double timeLimit, int)
//{
//	Timer t;
//	t.StartTimer();
//	
//	if (e->perfectEval(s) && e->canEval(s))
//	{
//		return PickBestMove(cc, s);
//	}
//	
//	RenewDepthEstimates(currState);
//	
//	samples = 0;
//	playoutLength = 0;
//	this->cc = cc;
//	tree.clear();
//	tree.resize(1);
//	tree[0].toMove = s.toMove;
//	tree[0].p1Dist = rootDepth1;
//	tree[0].p0Dist = rootDepth0;
//	this->currState = s;
//	Expand(0);
//	if (initFromDB)
//		InitFromDB(0, currState);
//	
//	while (t.GetElapsedTime() < timeLimit)
//	{
//		samples++;
//		this->currState = s;
//		SelectLeaf(0);
//	}
//	UCTDBNode &root = tree[0];
//	int best = root.firstChild;
//	for (int x = root.firstChild; x < root.firstChild + root.numChildren; x++)
//	{
//		std::cerr << tree[x].m << " " << tree[x].numSamples << " samples; avg: ";
//		std::cerr << tree[x].totalPayoff / tree[x].numSamples << "\n";
//		if (tree[x].totalPayoff / tree[x].numSamples >
//			tree[best].totalPayoff / tree[best].numSamples)
//		{
//			best = x;
//		}
//	}
//	value = tree[best].totalPayoff / tree[best].numSamples;
//	std::cerr << GetName() << " taking action " << tree[best].m << "\n";
//	std::cerr << "Tree size: " << tree.size() << " Avg playout length: " << playoutLength / samples << "\n";
//	return tree[best].m.clone(*cc);
//}
//
//template <class playoutModule, class evalModule>
//const char *UCTDB<playoutModule, evalModule>::GetName()
//{
//	name = "UCTDB-";
//	name += "C"+std::to_string(int(C))+"-";
//	name += "pd"+std::to_string(playOutDepth)+"-";
//	name += "el"+std::to_string(treeExpansionLimit)+"-";
//	if (initFromDB)
//		name += "initdb-";
//	name += m->GetName();
//	name += "-";
//	name += e->GetName();
//	return name.c_str();
//}
//
//template <class playoutModule, class evalModule>
//double UCTDB<playoutModule, evalModule>::SelectLeaf(uint32_t node)
//{
//	if (node != 0)
//		cc->ApplyMove(currState, &tree[node].m);
//	
//	double payoff;
//	tree[node].numSamples++;
//	if (IsLeaf(node))
//	{
//		if (tree[node].numSamples > treeExpansionLimit)
//		{
//			Expand(node);
//		}
//		payoff = DoPlayout(node);
//	}
//	else {
//		payoff = SelectLeaf(SelectBestChild(node));
//	}
//	if (tree[node].toMove == tree[0].toMove)
//	{
//		tree[node].totalPayoff += payoff;
//	}
//	else {
//		tree[node].totalPayoff -= payoff;
//	}
//	return payoff;
//}
//
//template <class playoutModule, class evalModule>
//uint32_t UCTDB<playoutModule, evalModule>::SelectBestChild(uint32_t node)
//{
//	UCTDBNode &n = tree[node];
//	int best = n.firstChild;
//	double bestVal = std::numeric_limits<double>::lowest();
//	for (int x = n.firstChild; x < n.firstChild + n.numChildren; x++)
//	{
//		if (tree[x].numSamples == 0)
//			return x;
//		double val = GetUCBVal(x, node);
//		if (val > bestVal)
//		{
//			best = x;
//			bestVal = val;
//		}
//	}
//	return best;
//}
//
//template <class playoutModule, class evalModule>
//double UCTDB<playoutModule, evalModule>::GetUCBVal(uint32_t node, uint32_t parent)
//{
//	double val = tree[node].totalPayoff/tree[node].numSamples;
//	val += C*sqrt(log(tree[parent].numSamples)/tree[node].numSamples);
//	return val;
//}
//
//
//template <class playoutModule, class evalModule>
//double UCTDB<playoutModule, evalModule>::DoPlayout(uint32_t node)
//{
//	int depth = 0;
//	while (true)
//	{
//		playoutLength++;
//		if (cc->Done(currState))
//		{
//			return e->eval(cc, currState, tree[0].toMove);
//		}
//		if ((depth > playOutDepth || e->perfectEval(currState)) && e->canEval(currState))
//		{
//			// TODO: add
//			return e->eval(cc, currState, tree[0].toMove);
//		}
//		CCMove *move = m->GetNextAction(cc, currState);
//		cc->ApplyMove(currState, move);
//		cc->freeMove(move);
//		depth++;
//	}
//}
//
//template <class playoutModule, class evalModule>
//bool UCTDB<playoutModule, evalModule>::IsLeaf(uint32_t node)
//{
//	return cc->Done(currState) || tree[node].numChildren == 0;
//}
//
//
//template <class playoutModule, class evalModule>
//void UCTDB<playoutModule, evalModule>::Expand(uint32_t node)
//{
//	if (cc->Done(currState))
//		return;
//	
//	CCMove *m;
//	//	if (node == 0)
//	//		m = cc->getMoves(currState);
//	//	else
//	m = cc->getMovesForward(currState);
//	tree[node].firstChild = static_cast<uint32_t>(tree.size());
//	tree[node].numChildren = m->length();
//	tree.resize(tree.size()+m->length());
//	uint32_t curr = tree[node].firstChild;
//	for (CCMove *t = m; t; t = t->next)
//	{
//		tree[curr].numChildren = 0;
//		tree[curr].firstChild = -1;
//		tree[curr].numSamples = 0;
//		tree[curr].toMove = currState.toMove;
//		tree[curr].m = *t;
//		tree[curr].m.next = 0;
//		tree[curr].totalPayoff = 0;
//		tree[curr].p0Dist = tree[node].p0Dist;
//		tree[curr].p1Dist = tree[node].p1Dist;
//
//		int depth;
//		cc->ApplyMove(currState, t);
//		db.GetRawDepth(currState, tree[node].toMove, depth);		cc->UndoMove(currState, t);
//		while (depth < tree[curr].p1Dist-10)
//			depth+=15;
//
//		curr++;
//	}
//	cc->freeMove(m);
//}
//
//template <class playoutModule, class evalModule>
//void UCTDB<playoutModule, evalModule>::InitFromDB(uint32_t node, CCState &s)
//{
//	const int count = 20;
//	int dthem = GetDepth("/data/cc/rev/", s, 1-s.toMove);
//	uint32_t first = tree[node].firstChild;
//	for (uint32_t c = 0; c < tree[node].numChildren; c++)
//	{
//		cc->ApplyMove(s, &tree[first+c].m);
//		int val = GetDepth("/data/cc/rev/", s, 1-s.toMove);
//		cc->UndoMove(s, &tree[first+c].m);
//		tree[first+c].totalPayoff += (dthem-val)*count;
//		tree[first+c].numSamples += count;
//	}
//}
//
//
//#endif
