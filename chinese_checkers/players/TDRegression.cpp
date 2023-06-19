//
//  TDRegression.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/18/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include "TDRegression.h"

double TDEval::eval(CCheckers *cc, CCState &s, int whichPlayer)
{
	GetFeatures(s, features, whichPlayer);
	double result = r.test(features);
	return result;
}

void TDEval::Train(CCheckers *cc, const std::vector<CCMove *> &trace)
{
	CCState s;
	cc->Reset(s);
	for (auto m : trace)
	{
		cc->ApplyMove(s, m);
	}
	// Game should be done at the end of the trace
	assert(cc->Done(s));
	int winner = cc->Winner(s);
	assert(winner != -1);
		
	double winReward = 1.0;
	double loseReward = -1.0;
	// Step backwards through the code training
	for (int x = int(trace.size()-1); x >= 0; x--)
	{
		cc->UndoMove(s, trace[x]);
		GetFeatures(s, features, s.toMove);
		if (s.toMove == winner)
		{
			r.setRate(0.1/features.size());
			r.train(features, winReward);
			winReward = lambda*winReward + (1-lambda)*r.test(features);
		}
		else {
			r.setRate(0.1/features.size());
			r.train(features, loseReward);
			loseReward = lambda*loseReward + (1-lambda)*r.test(features);
		}
	}
}

void TDEval::Print()
{
	auto w = r.GetWeights();
	for (int y = 0; y < 81; y++)
	{
		printf("%1.3f ", w[y]);
	}
	printf("\n");
}

void TDEval::GetFeatures(const CCState &s, std::vector<int> &f, int who)
{
	features.resize(0);
	features.resize(10*2);
	for (int x = 0; x < NUM_PIECES; x++)
	{
		if (who == 0)
		{
			features[x] = s.pieces[who][x];
			features[x+NUM_PIECES] = 81+80-s.pieces[1-who][x];
		}
		else {
			features[x] = 80-s.pieces[who][x];
			features[x+NUM_PIECES] = 81+s.pieces[1-who][x];
		}
	}
}


CCMove *TDRegression::GetRandom(CCheckers *cc, CCMove *m)
{
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

void TDRegression::GetFeatures(const CCState &s, std::vector<int> &features, int who)
{
	features.resize(0);
	features.resize(NUM_PIECES*2);
	for (int x = 0; x < NUM_PIECES; x++)
	{
		if (who == 0)
		{
			features[x] = s.pieces[who][x];
			features[x+NUM_PIECES] = 81+80-s.pieces[1-who][x];
		}
		else {
			features[x] = 80-s.pieces[who][x];
			features[x+NUM_PIECES] = 81+s.pieces[1-who][x];
		}
	}
}

CCMove *TDRegression::GetNextAction(CCheckers *cc, const CCState &s, double &bestVal, double timeLimit, int)
{
	CCState state = s;
	double val = random()%100000;
	val /= 100000;
	if (val < epsilon) // select random move
	{
		return GetRandom(cc, cc->getMoves(s));
	}
	CCMove *m = cc->getMovesForward(s);
	CCMove *best = 0;
	bestVal = -1000;
	//std::vector<int> features;
	for (CCMove *t = m; t; t = t->next)
	{
		if (cc->distance(t->from, cc->getGoal(s.toMove)) == cc->distance(t->to, cc->getGoal(s.toMove)))
			continue;
		
		cc->ApplyMove(state, t);
		GetFeatures(state, features, s.toMove);
		double value = r.test(features);
		if (value > bestVal && best == 0)
		{
			bestVal = value;
			best = t;
		}
		cc->UndoMove(state, t);
	}
	if (best == 0) // only sideways moves
	{
		return GetRandom(cc, m);
	}
	best = best->clone(*cc);
	cc->freeMove(m);
	return best;
}

void TDRegression::Train(CCheckers *cc, const std::vector<CCMove *> &trace)
{
	CCState s;
	cc->Reset(s);
	for (auto m : trace)
	{
		cc->ApplyMove(s, m);
	}
	// Game should be done at the end of the trace
	assert(cc->Done(s));
	int winner = cc->Winner(s);
	assert(winner != -1);

	std::vector<int> features;

	double winReward = 1.0;
	double loseReward = -1.0;
	// Step backwards through the code training
	for (int x = int(trace.size()-1); x >= 0; x--)
	{
		cc->UndoMove(s, trace[x]);
		GetFeatures(s, features, s.toMove);
		if (s.toMove == winner)
		{
			r.train(features, winReward);
			winReward = lambda*winReward + (1-lambda)*r.test(features);
		}
		else {
			r.train(features, loseReward);
			loseReward = lambda*loseReward + (1-lambda)*r.test(features);
		}
	}
}

void TDRegression::Print()
{
	auto w = r.GetWeights();
	for (int x = 0; x < 2; x++)
	{
		if (x == 0)
			printf("  Us: ");
		if (x == 1)
			printf("Them: ");
		if (x == 2)
			printf("Blnk: ");
		for (int y = 0; y < 81; y++)
		{
			printf("%1.3f ", w[y+81*x]);
		}
		printf("\n");
	}
}
