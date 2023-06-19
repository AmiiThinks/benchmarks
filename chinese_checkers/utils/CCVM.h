//
//  CCVM.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 2/18/18.
//  Copyright © 2018 NS Software. All rights reserved.
//


//~7 million first piece buckets
//
//~53 MB virtual table
//
//Each pointer either points to:
//
//1. Nothing (at the start)
//2. A pointer to a small memory chunk (1 kb - 4k states)
//3. A pointer to a large memory chunk  (~1.5MB - full 2nd player bucket)
//
//Memory budget:
//
//1000 2nd player buckets (1000x1.5MB = 1.5GB)
//7 million 1st player buckets (7000000x4k = 28GB)
//
//Additional in-memory cache
//(28GB?)
//
//16 bits wins from start
//16 bits losses from end


#ifndef CCVM_h
#define CCVM_h

#include <stdio.h>
#include "CCRankings.h"

struct VMEntry {
	//uint8_t *data; // Either a full or partial chunk of data
	uint16_t bigBucketOffset;
//	uint16_t bigBucketTimeStamp;
	//uint16_t fullData; // flag telling us whether the data is full or partial
	uint16_t smallBucketOffset; // offset in small data in smallDataSize chunks
	uint8_t dirty;
	//	uint16_t winOffset; // how many win states have been stored in the cache
//	uint16_t lossOffset; // how many loss states have been stored in the cache
};

// A Chinese Checkers-specific vm system.
class CCVM {
public:
	// maxR1 and maxR2 are in entries - actual storage is 2 bits per entry
	CCVM(const char *filename, int64_t maxR1, int64_t maxR2);
	~CCVM();
	uint64_t Get(uint64_t r1, uint64_t r2);
	void Set(uint64_t r1, uint64_t r2, uint64_t val);
	bool SetIf0(uint64_t r1, uint64_t r2, uint64_t val);
	// Indicates that reads will all come from this (address?) next
	//void SetNextR1(int64_t index);
	//uint64_t LockedGet(uint64_t r1, uint64_t r2);

	void Flush();
	bool runDetailedValidation;
private:
	void LoadBigBucket(int group);
	void UnloadBigBucket(int group);
	void SyncBigBucket(int group); // temporary for verification purposes
	void SyncSmallBucket(int group); // temporary for verification purposes
	void ValidateMemory();
	void ValidateMemoryBucket(int bucket);
	void ValidateSmallBucketVsDisk(int x);
	void LoadSmallBucket(int group, int32_t bucketOffset);
	void UnloadSmallBucket(int group);
	int32_t GetFreeBigBucket();
	int32_t nextBigBucket;
	CCPSRank12 r;
	VMEntry *memory;
	int64_t fullBucketSizeInEntries;
	int64_t smallBucketSizeInEntries;
	int64_t cacheBucketSizeInEntries;

	int64_t fullBucketSizeInBytes;
	int64_t smallBucketSizeInBytes;
	int64_t cacheBucketSizeInBytes;

	uint8_t *bigBucketMemory;
	uint8_t *smallBucketMemory;

	int32_t currentTimeStamp;
	
	FILE *f;
	// For now we keep the file in memory, pretending it is on disk
//	uint8_t *disk;
//	uint8_t *validation;
	int64_t maxR1, maxR2;
};

#endif /* CCVM_hpp */
