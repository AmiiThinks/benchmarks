//
//  PrioritySolver.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 7/23/11.
//  Copyright 2011 University of Denver. All rights reserved.
//
#include <stdint.h>

void PrioritySolver(std::vector<bool> &proof, const char *outputFile, bool stopAfterWin = true);
void BuildSASDistance(std::vector<uint8_t> &d);
void BuildSASDistance(std::vector<uint8_t> &d, std::vector<uint64_t> &startStates);
