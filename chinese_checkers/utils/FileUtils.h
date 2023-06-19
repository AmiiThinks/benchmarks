//
//  FileUtils.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 7/27/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include <vector>
#include <stdio.h>

void WriteData(std::vector<bool> &data, const char *fName);
void ReadData(std::vector<bool> &data, const char *fName);
