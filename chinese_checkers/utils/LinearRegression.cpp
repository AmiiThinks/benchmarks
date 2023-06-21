/*
 *  LinearRegression.cpp
 *  hog2
 *
 *  Created by Nathan Sturtevant on 6/1/06.
 *  Copyright 2006 Nathan Sturtevant. All rights reserved.
 *
 */


#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include "LinearRegression.h"

LinearRegression::LinearRegression(int _inputs, double _rate)
{
	inputs = _inputs;
	rate = _rate;
	weights.resize(inputs+1);
}

void LinearRegression::train(std::vector<double> &input, double target)
{
	double output = test(input);
	for (int x = 0; x < input.size(); x++)
	{
		weights[x] += rate*(target-output)*input[x];
	}
	weights[input.size()] += rate*(target-output);
}

double LinearRegression::test(const std::vector<double> &input)
{
	double result = 0;
	for (int x = 0; x < input.size(); x++)
	{
		result += input[x]*weights[x];
	}
	result += weights.back();
	return result;
}

void LinearRegression::train(std::vector<int> &input, double target)
{
	double output = test(input);
	for (int x = 0; x < input.size(); x++)
	{
		weights[input[x]] += rate*(target-output);
	}
	weights[input.size()] += rate*(target-output);
}

double LinearRegression::test(const std::vector<int> &input)
{
	double result = 0;
	for (int x = 0; x < input.size(); x++)
	{
		result += weights[input[x]];
	}
	result += weights.back();
	return result;
}


void LinearRegression::Print()
{
	for (int x = 0; x <= inputs; x++)
	{
		printf("%1.3f  ", weights[x]);
	}
	printf("\n");
}
