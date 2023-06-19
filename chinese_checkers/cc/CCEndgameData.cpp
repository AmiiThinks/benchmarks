//
//  CCEndgameData.cpp
//  CC UCT
//
//  Created by Nathan Sturtevant on 4/22/15.
//  Copyright (c) 2015 Nathan Sturtevant. All rights reserved.
//

#include "CCEndgameData.h"
#include <string.h>
#include <assert.h>
#include <algorithm>
#include <string.h>
//std::vector<uint8_t*> rawData;

const int64_t maxFileEntries = 1<<30;

CCEndGameData::CCEndGameData(const char *prefix, int firstTwoPieceRank)
{
	int64_t entries = cc.getMaxSinglePlayerRank2();
	rawData.resize(entries);
	char fname[1024];
	assert(strlen(prefix)+32 < 1024);
	std::vector<int> histogram;
	
	for (int x = firstTwoPieceRank; x < rawData.size(); x++)
	{
		histogram.clear();
		histogram.resize(15);
		int64_t numEntries = cc.getMaxSinglePlayerRank2(x);
		for (int64_t y = 0; y < numEntries; y+=maxFileEntries)
		{
			sprintf(fname, "%sCC-%d-%d-b%d-f%lld.bin", prefix, NUM_SPOTS, NUM_PIECES, int(x), y/maxFileEntries);
			FILE *f = fopen(fname, "r");
			if (f == 0)
			{
				fprintf(stderr, "Unable to open '%s'; skipping\n", fname);
				break;
			}
			if (rawData[x] == 0)
			{
				rawData[x] = new uint8_t[(numEntries+1)/2];
				//printf("Allocating %llu bytes for %d (%p)\n", (numEntries+1)/2, x, rawData[x]);
			}
			
			fread(&rawData[x][y], sizeof(uint8_t), std::min(maxFileEntries/2, (numEntries-y+1)/2), f);
			fclose(f);
		}
		if (rawData[x])
		{
			//printf("Histogram for bucket %d:\n", x);
			for (int64_t y = 0; y < numEntries; y+=2)
			{
				uint8_t val = rawData[x][y/2];
				histogram[val&0xF]++;
				histogram[(val>>4)&0xF]++;
			}
//			for (int x = 0; x < 15; x++)
//				printf("%2d: %d\n", x, histogram[x]);
//			fflush(stdout);
		}
	}
}

CCEndGameData::~CCEndGameData()
{
	fprintf(stderr, "Cleaning up endgame data\n");
	for (auto &x : rawData)
	{
		if (x)
		{
			delete [] x;
			x = 0;
		}
	}
}

int CCEndGameData::LowerBound(const CCState &s, int who) const
{
	int inPlace = 0;
	int moves1 = 0;
	int common = 0;
	memset(dist, 0, 17);
	int far = 0;
	int far2 = 0;
	int low = 0, high = 0;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		int next = cc.distance(s.pieces[who][x], cc.getGoal(who));
		dist[next]++;
		if (next > 12)
			far++;
		if (next > 9)
			far2++;
		if (next > 4)
		{
			int x1, y1;
			cc.toxy(s.pieces[who][x], x1, y1);
			if (x1-5 < 5)
				low++;
			if (x1-5 > 11)
				high++;
		}
	}
	moves1 += far/2;
	if (far == 4)
		moves1--;
	if (far2 > 5)
		moves1+=2;
	// test
	for (int x = 0; x < 17 && inPlace != NUM_PIECES; x++)
	{
		if (x < 4)
		{
			inPlace += dist[x];
		}
		else {
			if (dist[x] == 0)
			{
				// at least one move to fill in empty row for later jumping
				moves1++;
			}
			else {
				common += dist[x];
				inPlace += dist[x];
			}
		}
	}
	if (low && high)
	{
		moves1+=2;
		if (low > 2 || high > 2)
			moves1 += 2;
	}
	return common+moves1;
}

bool CCEndGameData::GetDepth(const CCState &s, int who, int &depth) const
{
	bool res = GetRawDepth(s, who, depth);
	int lb = LowerBound(s, who);
	while (depth < lb)
		depth += 15;
	return res;
}

bool CCEndGameData::GetRawDepth(const CCState &s, int who, int &depth) const
{
	CCState flipped;
	int64_t r1, r2;
	if (who == 0)
		cc.rankPlayer(s, 0, r1, r2);
	else
		cc.rankPlayerFlipped(s, 1, r1, r2);
	uint8_t value;
	if (rawData[r1] != 0)
	{
		int64_t offset = r2/2;
		uint8_t* ptr = rawData[r1];
		value = ptr[offset];
		if ((r2%2) == 0)
		{
			depth = (value>>0)&0xF;
			return true;
		}
		else {
			depth = (value>>4)&0xF;
			return true;
		}
	}

	if (who == 0)
	{
		cc.FlipPlayer(s, flipped, 0);
		cc.rankPlayer(flipped, 0, r1, r2);
	}
	else {
		cc.FlipPlayer(s, flipped, 1);
		cc.rankPlayerFlipped(flipped, 1, r1, r2);
	}
	
	if (rawData[r1] != 0)
	{
		value = rawData[r1][r2/2];
		if ((r2%2) == 0)
		{
			depth = (value>>0)&0xF;
			return true;
		}
		else {
			depth = (value>>4)&0xF;
			return true;
		}
	}
	return false;
}

