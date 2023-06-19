//
//  CCUtils.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/20/13.
//
//

#ifndef __Solve_Chinese_Checkers__CCUtils__
#define __Solve_Chinese_Checkers__CCUtils__

#include <iostream>
#include <vector>
#include <stdint.h>
#include "CCheckers.h"

//const char *dataPrefix = "/data/cc/rev/";
const char* const dataPrefix = "/Users/nathanst/Development/cc/data/";
const char* const fullDataPrefix = "/data/cc/rev/";


struct bucketStructure {
	int64_t firstGroup, lastGroup;
	int64_t whenLoad, whenWrite;
	size_t memoryAddress;
	int numNeighbors, tmpValue;
	std::vector<bool> touchesBucket;
	std::vector<bool> touchedGroups;
	std::vector<int> touchedGroupsID;
};

struct bucketInfo {
	bool unused;
	//bool special;
	int bucketID;
	int64_t numEntries;
	int64_t bucketOffset;
};

struct bucketChanges {
	bucketChanges() { 	pthread_mutex_init(&lock, NULL); }
	bool updated;
	int currDepthWritten;
	int lastDepthWritten;
	int64_t remainingEntries;
	std::vector<bool> changes;
	std::vector<bool> roundChanges;
	std::vector<bool> nextChanges;
	std::vector<bool> coarseClosed;
	pthread_mutex_t lock;
};

struct bucketData {
	int64_t theSize;
#ifndef DISK
	std::vector<uint8_t> data;
#endif
};

int GetDepth(const char *prefix, CCState &s, int who);
int GetBackPieceAdvancement(const CCState &s, int who);

void InitTwoPieceData(std::vector<bucketInfo> &data, uint64_t maxBucketSize, int openSize, bool symmetry);
void InitBuckets(uint64_t maxBucketSize, std::vector<bucketChanges> &twoPieceChanges,
				 std::vector<bucketData> &buckets, int openSize, bool symmetry, bool extraCoarseData = true);
int64_t InitTwoPieceStructure(const std::vector<bucketInfo> &data, int numBuckets, std::vector<bucketStructure> &structure,
						   std::vector<int> &readOrder, bool symmetry, int count);

void MakeSVG(const CCState &s, const char *output);

#endif /* defined(__Solve_Chinese_Checkers__CCUtils__) */
