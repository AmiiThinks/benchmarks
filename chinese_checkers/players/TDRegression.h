//
//  TDRegression.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/18/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef TD_REGRESSION_H
#define TD_REGRESSION_H

#include <stdio.h>
#include <assert.h>
#include <string>
#include "CCheckers.h"
#include "Timer.h"
#include "Player.h"
#include "LinearRegression.h"

class TDEval {
public:
	TDEval(double e, double l) :epsilon(e), lambda(l), r(81*2, 0.01) {  }
	void RootState(const CCState &s) {}
	double eval(CCheckers *cc, CCState &s, int whichPlayer);
	void Train(CCheckers *cc, const std::vector<CCMove *> &trace);
	void Print();
	const char *GetName()
	{ return "TD"; }
	bool perfectEval(const CCState &s) { return false; }
	bool canEval(const CCState &) { return true; }
	void SetWeights(const std::vector<double> &w)
	{ r.SetWeights(w); }
private:
	void GetFeatures(const CCState &s, std::vector<int> &features, int who);
	LinearRegression r;
	double epsilon, lambda;
	std::vector<int> features;
};

class TDRegression : public Player {
public:
	TDRegression(double e, double l) :epsilon(e), lambda(l), r(81*2, 0.001) {  }
	CCMove *GetNextAction(CCheckers *cc, const CCState &s, double &, double timeLimit, int depthLimit = 100000);
	void Train(CCheckers *cc, const std::vector<CCMove *> &trace);
	void Print();
	virtual const char *GetName()
	{
		return "TD-Regression";
	}
	void Reset() {}
	TDEval GetTDEval()
	{
		TDEval tde(epsilon, lambda);
		tde.SetWeights(r.GetWeights());
		return tde;
	}
private:
	void GetFeatures(const CCState &s, std::vector<int> &features, int who);
	CCMove *GetRandom(CCheckers *cc, CCMove *);
	LinearRegression r;
	double epsilon, lambda;
	std::vector<int> features;
};

#endif
