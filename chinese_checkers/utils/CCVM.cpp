//
//  CCVM.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 2/18/18.
//  Copyright © 2018 NS Software. All rights reserved.
//

#include "CCVM.h"
#include <cassert>
#include <cstring>

//CCPSRank12 r;
//VMEntry *memory;
//int64_t fullDataSizeInBytes;
//int64_t smallDataSizeInBytes;
//int64_t cacheDataSizeInBytes;
//int8_t *bigBuckets;
//int8_t *smallBuckets;
//int32_t *localCache;

const uint16_t kNoInit = 0xFFFF;
const int64_t numBigBuckets = 1024*16;

CCVM::CCVM(const char *filename, int64_t _maxR1, int64_t _maxR2)
:maxR1(_maxR1), maxR2(_maxR2)
{
	fullBucketSizeInEntries = maxR2;
	smallBucketSizeInEntries = 4096; // 2bit entries = 1024 bytes
	cacheBucketSizeInEntries = 256; // 32bit entries = 1024 bytes
	
	fullBucketSizeInBytes = (fullBucketSizeInEntries+3)/4;
	smallBucketSizeInBytes = (smallBucketSizeInEntries+3)/4;
	cacheBucketSizeInBytes = cacheBucketSizeInEntries*4;

	memory = new VMEntry[maxR1];
	// maxR1
	bigBucketMemory = new uint8_t[fullBucketSizeInBytes*numBigBuckets];
	smallBucketMemory = new uint8_t[smallBucketSizeInBytes*maxR1];
//	localCacheMemory = new int32_t[cacheBucketSizeInEntries*maxR1];

	// Everything that will be on disk eventually
//	disk = new uint8_t[maxR1*fullBucketSizeInBytes];
//	validation = new uint8_t[maxR1*fullBucketSizeInBytes];
//	memset(disk, 0, maxR1*fullBucketSizeInBytes);
//	memset(validation, 0, maxR1*fullBucketSizeInBytes);

	for (int64_t x = 0; x < maxR1; x++)
	{
		memory[x].bigBucketOffset = kNoInit;
//		memory[x].bigBucketTimeStamp = 0;
		memory[x].smallBucketOffset = kNoInit; // offset in small data in smallDataSize chunks
//		memory[x].winOffset = 0; // how many win states have been stored in the cache
//		memory[x].lossOffset = 0; // how many loss states have been stored in the cache
		memory[x].dirty = false;
	}

	nextBigBucket = 0;

	// TODO: Need data structure to hand out the big buckets
	// for now everyone gets a big bucket
	memset(bigBucketMemory, 0, fullBucketSizeInBytes*numBigBuckets);
	memset(smallBucketMemory, 0, smallBucketSizeInBytes*maxR1);
//	memset(localCacheMemory, 0, cacheBucketSizeInEntries*sizeof(uint32_t)*maxR1);

	currentTimeStamp = 1;
	// TODO: Initialize disk to be empty
	printf("Opening '%s'\n", filename);
	f = fopen(filename, "w+");
	if (f == 0)
	{
		printf("Open failed\n");
		exit(0);
	}
	uint32_t totalFileSize = fullBucketSizeInBytes*maxR1;
	printf("%llu r1, %llu r2. Total bytes; %d\n", maxR1, maxR2, totalFileSize);
	uint8_t buffer[1024];
	memset(buffer, 0, 1024);
	fseek(f, 0, SEEK_SET);
	for (int x = 0; x < (totalFileSize+1023)/1024; x++)
	{
		fwrite(buffer, 1, 1024, f);
	}
}

CCVM::~CCVM()
{
	// close files
	
	// free memory
	delete [] memory;
	delete [] bigBucketMemory;
	delete [] smallBucketMemory;
//	delete [] localCacheMemory;
	
	//delete [] disk;
	//fclose(f);
}

void CCVM::LoadBigBucket(int group)
{
	// TODO: Can just load the big bucket and copy the values over instead of copying back and forth
	UnloadSmallBucket(group);

	if (memory[group].bigBucketOffset != kNoInit)
		return;
//	printf("Loading BIG %d\n", group);
	memory[group].bigBucketOffset = GetFreeBigBucket();
	fseek(f, group*fullBucketSizeInBytes, SEEK_SET);
	fread(&bigBucketMemory[memory[group].bigBucketOffset*fullBucketSizeInBytes],
		  1, fullBucketSizeInBytes, f);
//	memcpy(&bigBucketMemory[memory[group].bigBucketOffset*fullBucketSizeInBytes], // the big bucket offset determines our location in cache
//		   &disk[group*fullBucketSizeInBytes], // The group determines our location on disk
//		   fullBucketSizeInBytes);
}

void CCVM::UnloadBigBucket(int group)
{
	if (memory[group].bigBucketOffset != kNoInit)
	{
		fseek(f, group*fullBucketSizeInBytes, SEEK_SET);
		fwrite(&bigBucketMemory[memory[group].bigBucketOffset*fullBucketSizeInBytes],
			  1, fullBucketSizeInBytes, f);
//		memcpy(&disk[group*fullBucketSizeInBytes],
//			   &bigBucketMemory[memory[group].bigBucketOffset*fullBucketSizeInBytes],
//			   fullBucketSizeInBytes);
		memory[group].bigBucketOffset = kNoInit;
	}
}

void CCVM::LoadSmallBucket(int group, int32_t bucketOffset)
{
	assert(memory[group].bigBucketOffset == kNoInit);
	bucketOffset /= smallBucketSizeInEntries;
	// already loaded - done!
	if (memory[group].smallBucketOffset == bucketOffset)
		return;
	UnloadSmallBucket(group);
//	printf("Loading small %d-%d\n", group, bucketOffset);
	memory[group].smallBucketOffset = bucketOffset;
	uint32_t amountToCopy = std::min(smallBucketSizeInBytes, fullBucketSizeInBytes-bucketOffset*smallBucketSizeInBytes);
	fseek(f, group*fullBucketSizeInBytes+memory[group].smallBucketOffset*smallBucketSizeInBytes, SEEK_SET);
	fread(&smallBucketMemory[group*smallBucketSizeInBytes], 1, amountToCopy, f);
//	memcpy(&smallBucketMemory[group*smallBucketSizeInBytes],
//		   &disk[group*fullBucketSizeInBytes+memory[group].smallBucketOffset*smallBucketSizeInBytes],
//		   amountToCopy);
	memory[group].dirty = false;
}

void CCVM::UnloadSmallBucket(int group)
{
	if (memory[group].smallBucketOffset != kNoInit)
	{
		if (memory[group].dirty)
		{
			memory[group].dirty = false;
			uint32_t amountToCopy = std::min(smallBucketSizeInBytes, fullBucketSizeInBytes-memory[group].smallBucketOffset*smallBucketSizeInBytes);
			fseek(f, group*fullBucketSizeInBytes+memory[group].smallBucketOffset*smallBucketSizeInBytes, SEEK_SET);
			fwrite(&smallBucketMemory[group*smallBucketSizeInBytes], 1, amountToCopy, f);
//			memcpy(&disk[group*fullBucketSizeInBytes+memory[group].smallBucketOffset*smallBucketSizeInBytes],
//				   &smallBucketMemory[group*smallBucketSizeInBytes],
//				   amountToCopy);
		}
		memory[group].smallBucketOffset = kNoInit;
	}
}

int32_t CCVM::GetFreeBigBucket()
{
	if (nextBigBucket < numBigBuckets)
	{
		int32_t bucket = nextBigBucket;
		nextBigBucket++;
		return bucket;
	}
	// For now, remove the earliest one
	for (int x = 0; x < maxR1; x++)
	{
		if (memory[x].bigBucketOffset != kNoInit)
		{
			int32_t bucket = memory[x].bigBucketOffset;
			UnloadBigBucket(x);
			return bucket;
		}
	}
	assert(!"Couldn't find any buckets");
	return -1;
}

void CCVM::Flush()
{
	// Write data to memory, reset time stamps
	nextBigBucket = 0;
	for (int x = 0; x < maxR1; x++)
	{
//		memory[x].bigBucketTimeStamp = 0;
		UnloadBigBucket(x);
		UnloadSmallBucket(x);
	}
}

uint64_t CCVM::Get(uint64_t r1, uint64_t r2)
{
	uint64_t r2Byte = r2/4;
	uint64_t r2Bit = r2&0x3;
	// in big cache
	if (memory[r1].bigBucketOffset != kNoInit)
	{
		uint64_t result = ((bigBucketMemory[memory[r1].bigBucketOffset*fullBucketSizeInBytes+r2Byte])>>(2*r2Bit))&0x3;
		return result;
	}
	
	LoadSmallBucket(r1, r2);
	// if it is in the small cache, return
	assert(r2/smallBucketSizeInEntries == memory[r1].smallBucketOffset);
	
	uint64_t loc = r2%smallBucketSizeInEntries;
	uint64_t result = (smallBucketMemory[r1*smallBucketSizeInBytes+loc/4]>>(2*(loc&0x3)))&0x3;
	return result;
}

void CCVM::Set(uint64_t r1, uint64_t r2, uint64_t val)
{
	// If in large cache, write immediately
	if (memory[r1].bigBucketOffset != kNoInit)
	{
		// One less operation to:
		// shift old
		// xor new
		// mask result
		// shift result
		// xor in
		// as opposed to: shift/invert mask (2), apply mask (1), shift/mask new (2), or in new value (1)
		uint64_t r2Byte = r2/4;
		uint64_t r2Bit = (r2&0x3)*2;
		uint8_t tmp = bigBucketMemory[memory[r1].bigBucketOffset*fullBucketSizeInBytes+r2Byte];
		tmp = (tmp&(~(0x3<<r2Bit)))|((val&0x3)<<r2Bit);
		bigBucketMemory[memory[r1].bigBucketOffset*fullBucketSizeInBytes+r2Byte] = tmp;
		return;
	}
	// TODO: support other caches
	//assert(!"Write data should be in large cache");

	{
		LoadSmallBucket(r1, r2);
		assert(r2/smallBucketSizeInEntries == memory[r1].smallBucketOffset);

		uint64_t loc = r2%smallBucketSizeInEntries;
		uint64_t r2Byte = loc/4;
		uint64_t r2Bit = (loc&0x3)*2;
		uint8_t tmp = (smallBucketMemory[r1*smallBucketSizeInBytes+r2Byte]);
		tmp = (tmp&(~(0x3<<r2Bit)))|((val&0x3)<<r2Bit);
		smallBucketMemory[r1*smallBucketSizeInBytes+r2Byte] = tmp;
		memory[r1].dirty = true;
	}
	// If in small cache, write immediately
	// write to buffer
	// if buffer almost full, load and flush buffer

}

// Returns true if set
bool CCVM::SetIf0(uint64_t r1, uint64_t r2, uint64_t val)
{
	LoadBigBucket(r1);
	// If in large cache, write immediately
	assert(memory[r1].bigBucketOffset != kNoInit);

	// One less operation to:
	// shift old / xor new / mask result / shift result / xor in
	// as opposed to: shift/invert mask (2), apply mask (1), shift/mask new (2), or in new value (1)
	uint64_t r2Byte = r2/4;
	uint64_t r2Bit = (r2&0x3)*2;
	uint8_t tmp = bigBucketMemory[memory[r1].bigBucketOffset*fullBucketSizeInBytes+r2Byte];
	if (((tmp>>r2Bit)&0x3) != 0)
		return false;
	tmp = (tmp&(~(0x3<<r2Bit)))|((val&0x3)<<r2Bit);
	bigBucketMemory[memory[r1].bigBucketOffset*fullBucketSizeInBytes+r2Byte] = tmp;
	return true;
}

