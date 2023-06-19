//
//  VirtualMemory.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 10/12/12.
//
//

#include "VirtualMemory.h"
#include <assert.h>

const int pagesPerLock = 9; // actually, 2^

VirtualMemory::VirtualMemory(uint64_t memoryPageSize, int coarseSize, uint64_t maxEntries, uint64_t virtualMemorySize)
:bits((virtualMemorySize|memoryPageSize-1)+1), values(maxEntries), tlb((maxEntries+memoryPageSize-1)/memoryPageSize)
{
	virtualMemorySize = (virtualMemorySize|memoryPageSize-1)+1;
	//((maxRank|8191)+1)/2
	coarseOpenSize = coarseSize;
	for (unsigned int x = 0; x < tlb.size(); x++)
		tlb[x] = -1;
	pageSize = memoryPageSize;
	assert(0 == (pageSize&(pageSize-1))); // page size is power of 2
	bitsOffset = 0;
	assert(0 == virtualMemorySize%pageSize);

	coarseNextDepth.resize((maxEntries+coarseOpenSize-1)/coarseOpenSize);
	coarseCurrDepth.resize((maxEntries+coarseOpenSize-1)/coarseOpenSize);
	coarseUnwritten.resize((maxEntries+coarseOpenSize-1)/coarseOpenSize);
	
	int64_t tmp = pageSize;
	for (pageBits = -1; tmp; pageBits++)
	{
		tmp>>=1;
	}
	assert(pageSize == (1<<pageBits));
	printf("%d bits for each page\n", pageBits);

	printf("Memory required:\n");
	printf("\tIndex (TLB):  %3llu MB\n", virtualMemorySize/pageSize/1024/1024);
	printf("\tVirt. Memory: %3llu MB\n", virtualMemorySize/1024/1024);
	printf("\tCoarse Open:  %3llu MB\n", 3*virtualMemorySize/coarseOpenSize/1024/1024);
	printf("Total: %llu MB\n", virtualMemorySize/pageSize/1024/1024+virtualMemorySize/1024/1024+3*virtualMemorySize/coarseOpenSize/1024/1024);
	printf("Locks: %lld\n", (maxEntries>>pageBits)>>(pagesPerLock+1));

	maxEntry = maxEntries;
	for (uint64_t x = 0; x < maxEntries; x++)
		values[x] = 255;
	locks.resize(((maxEntries>>pageBits)>>pagesPerLock)+1);
	for (unsigned int x = 0; x < locks.size(); x++)
		pthread_mutex_init(&locks[x], NULL);
	pthread_mutex_init(&vmLock, NULL);

	unstoredStates = false;
}

uint64_t VirtualMemory::NextStateAtDepth(uint64_t lastState, int depth)
{
	for (uint64_t x = lastState+1; x < maxEntry; x++)
	{
		if (coarseCurrDepth[x/coarseOpenSize] == false)
		{
			x += coarseOpenSize-1;
			continue;
		}
		if (values[x] == depth)
			return x;
	}
	return neg1;
}

uint64_t VirtualMemory::GetNumMemoryLocks()
{
	return locks.size();
}

uint64_t VirtualMemory::GetLockNumber(uint64_t location)
{
	return (location>>pageBits)>>pagesPerLock;
}

void VirtualMemory::Lock(uint64_t which)
{
	pthread_mutex_lock(&locks[which]);
}

void VirtualMemory::Unlock(uint64_t which)
{
	pthread_mutex_unlock(&locks[which]);
}

//void VirtualMemory::WriteDepth(uint64_t state, uint64_t source)
void VirtualMemory::WriteDepth(uint64_t location)
{
	bits[location] = true;
}

uint64_t VirtualMemory::GetVirtualAddress(uint64_t state, uint64_t source)
{
	uint64_t index = state>>pageBits;
	if (tlb[index] != -1)
		return (tlb[index]*pageSize)+(state&(pageSize-1));
	
	pthread_mutex_lock (&vmLock);
	tlb[index] = bitsOffset++;
	pthread_mutex_unlock (&vmLock);
	
	assert(bitsOffset < bits.size()>>pageBits);
	return tlb[index];
}

void VirtualMemory::GetVirtualAddress(std::vector<uint64_t> &states, std::vector<uint64_t> &sources, std::vector<uint64_t> &addresses)
{
	addresses.resize(states.size());
	bool locked = false;
	
	for (unsigned int x = 0; x < states.size(); x++)
	{
		uint64_t index = states[x]>>pageBits;
		int32_t val = tlb[index];
		if (val == -2)
		{
			addresses[x] = neg1;
			continue;
		}
		else if (val == -1)
		{
			if (!locked)
			{
				pthread_mutex_lock(&vmLock);
				locked = true;
				val = tlb[index];
			}
			if (val == -1)
			{
				if (bitsOffset == bits.size()/pageSize)
				{
					addresses[x] = neg1;
					coarseUnwritten[sources[x]/coarseOpenSize] = true;
					unstoredStates = true;
					continue;
				}
				else {
					val = bitsOffset;
					tlb[index] = bitsOffset++;
				}
			}
		}
		addresses[x] = (val*pageSize)+(states[x]&(pageSize-1));
	}
	if (locked)
	{
		pthread_mutex_unlock(&vmLock);
	}
	states.resize(0);
	sources.resize(0);
}


uint64_t VirtualMemory::FlushToDisk()
// TODO: don't write to memory blocks that were in memory last time(?)
// TODO: Use -2 to mark tlb entries which shouldn't be udpated this round
// TODO: Only clear tlb entries to -1 when there are no unstored states
{
	uint64_t statesWritten = 0;
	for (uint64_t x = 0; x < tlb.size(); x++)
	{
		if (tlb[x] == -1 || tlb[x] == -2)
			continue;
		for (uint64_t y = 0; y < pageSize; y++)
		{
			if (bits[static_cast<uint64_t>(tlb[x])*pageSize+y])
			{
//				printf("   Read bit %llu as true\n", static_cast<uint64_t>(tlb[x])*pageSize+y);
				if (values[x*pageSize+y] == 255)
				{
					values[x*pageSize+y] = currDepth;
					coarseNextDepth[(x*pageSize+y)/coarseOpenSize] = true;
					statesWritten++;
				}
				bits[tlb[x]*pageSize+y] = false;
			}
		}
//		tlb[x] = -1;
	}
	bitsOffset = 0;
	uint64_t count = 0;
	if (unstoredStates)
	{
		uint64_t lastCoarse = 0, currCoarse = 0;
		for (unsigned int x = 0; x < coarseCurrDepth.size(); x++)
		{
			lastCoarse += coarseCurrDepth[x]?1:0;
			currCoarse += coarseUnwritten[x]?1:0;
		}
		coarseCurrDepth = coarseUnwritten;
		coarseUnwritten.resize(0);
		coarseUnwritten.resize((maxEntry+coarseOpenSize-1)/coarseOpenSize);
//		unstoredStates = false;
		for (unsigned int x = 0; x < tlb.size(); x++)
		{
			if (tlb[x] != -1)
			{
				count++;
				tlb[x] = -2;
			}
		}
		printf(" * Unstored states; repeating iteration. %llu of %lu tlb entries marked as finished\n", count, tlb.size());
		printf(" * Coarse open reduced from %llu to %llu entries\n", lastCoarse, currCoarse);
		fflush(stdout);
	}
	else {
		coarseCurrDepth = coarseNextDepth;
		coarseNextDepth.resize(0);
		coarseNextDepth.resize((maxEntry+coarseOpenSize-1)/coarseOpenSize);
		unstoredStates = false;
		for (unsigned int x = 0; x < tlb.size(); x++)
		{
			tlb[x] = -1;
		}
	}
	
//	for (unsigned int x = 0; x < bits.size(); x++)
//	{
//		if (bits[x])
//		{
//			if (values[x] == 255)
//			{
//				values[x] = currDepth;
//				statesWritten++;
//			}
//		}
//		bits[x] = false;
//	}
	return statesWritten;
}

bool VirtualMemory::HadUnstoredStates()
{
	if (unstoredStates)
	{
		unstoredStates = false;
		return true;
	}
	return unstoredStates;
}
