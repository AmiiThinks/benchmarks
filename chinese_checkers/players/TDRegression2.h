//
//  TDRegression.h
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/18/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#ifndef TD_REGRESSION2_H
#define TD_REGRESSION2_H

#include <stdio.h>
#include <assert.h>
#include <string>
#include "CCheckers.h"
#include "Timer.h"
#include "Player.h"
#include "LinearRegression.h"

class TDEval2 {
public:
	TDEval2(double e, double l) :epsilon(e), lambda(l), r(81*2+81*81*2, 0.001) {  }
	void RootState(const CCState &s) {}
	double eval(CCheckers *cc, CCState &s, int whichPlayer);
	void Train(CCheckers *cc, const std::vector<CCMove *> &trace);
	void Print();
	const char *GetName()
	{ return "TD2"; }
	void SetWeights(const std::vector<double> &w)
	{ r.SetWeights(w); }
private:
	void GetFeatures(const CCState &s, std::vector<int> &features, int who);
	LinearRegression r;
	double epsilon, lambda;
	std::vector<int> features;
};

class TDRegression2 : public Player {
public:
	TDRegression2(double e, double l) :epsilon(e), lambda(l), r(81*2+81*81*2, 0.001) {  }
	CCMove *GetNextAction(CCheckers *cc, const CCState &s, double &, double timeLimit, int depthLimit = 100000);
	void Train(CCheckers *cc, const std::vector<CCMove *> &trace);
	void Print();
	virtual const char *GetName()
	{
		return "TD-Regression2";
	}
	void Reset() {}
	TDEval2 GetTDEval()
	{
		TDEval2 tde(epsilon, lambda);
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
