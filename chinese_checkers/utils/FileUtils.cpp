//
//  FileUtils.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 7/27/11.
//  Copyright 2011 University of Denver. All rights reserved.
//

#include "FileUtils.h"
#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void WriteData(std::vector<bool> &data, const char *fName)
{
	FILE *f = fopen(fName, "w+");
	fprintf(f, "%llu\n", (uint64_t)data.size());
	
	uint8_t next = 0;
	for (uint64_t x = 0; x < data.size(); x++)
	{
		next = (next<<1)|(data[x]?1:0);
		if (7 == x%8)
		{
			fwrite(&next, sizeof(uint8_t), 1, f);
			next = 0;
		}
	}
	fwrite(&next, sizeof(uint8_t), 1, f);
	fclose(f);
}

void ReadData(std::vector<bool> &data, const char *fName)
{
	FILE *f = fopen(fName, "r+");
	if (f == 0)
	{
		printf("Unable to open file: '%s' aborting\n", fName);
		exit(0);
	}
	uint64_t dataSize;
	fscanf(f, "%llu\n", &dataSize);
	data.resize(dataSize);
	
	uint8_t next = 0;
	uint64_t x;
	for (x = 8; x < data.size(); x+=8)
	{
		fread(&next, sizeof(uint8_t), 1, f);
		data[x-8+0] = (next>>7)&0x1;
		data[x-8+1] = (next>>6)&0x1;
		data[x-8+2] = (next>>5)&0x1;
		data[x-8+3] = (next>>4)&0x1;
		data[x-8+4] = (next>>3)&0x1;
		data[x-8+5] = (next>>2)&0x1;
		data[x-8+6] = (next>>1)&0x1;
		data[x-8+7] = (next>>0)&0x1;
	}
	fread(&next, sizeof(uint8_t), 1, f);
	for (uint64_t y = 0; x-8+y < data.size(); y++)
		data[x-8+y] = (next>>(7-y))&0x1;
	fclose(f);
}
