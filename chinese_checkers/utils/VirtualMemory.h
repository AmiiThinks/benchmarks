//
//  VirtualMemory.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 10/12/12.
//
//

#ifndef __Solve_Chinese_Checkers__VirtualMemory__
#define __Solve_Chinese_Checkers__VirtualMemory__

#include <iostream>
#include <vector>
#include <stdint.h>
#include <stdio.h>

const uint64_t neg1 = 0ull-1ull;

class VirtualMemory {
public:
	VirtualMemory(uint64_t memoryPageSize, int coarseOpenSize, uint64_t maxEntries, uint64_t virtualMemorySize);
	uint64_t NextStateAtDepth(uint64_t lastState, int depth);
	void WriteDepth(uint64_t location);
	uint64_t FlushToDisk();
	bool HadUnstoredStates();
	void SetCurrDepth(uint8_t d) { currDepth = d; }
	uint64_t GetVirtualAddress(uint64_t state, uint64_t source);
	void GetVirtualAddress(std::vector<uint64_t> &states, std::vector<uint64_t> &sources, std::vector<uint64_t> &addresses);
	
	uint64_t GetNumMemoryLocks();
	uint64_t GetLockNumber(uint64_t location);
	void Lock(uint64_t which);
	void Unlock(uint64_t which);
private:
	std::vector<bool> bits;
	std::vector<bool> coarseNextDepth, coarseCurrDepth, coarseUnwritten;
	std::vector<uint8_t> values;
	std::vector<int32_t> tlb;
	std::vector<pthread_mutex_t> locks;
	pthread_mutex_t vmLock;
	uint64_t maxEntry;
	uint8_t currDepth;
	uint64_t pageSize;
	int pageBits;
	int bitsOffset;
	int coarseOpenSize;
	bool unstoredStates;
};

#endif /* defined(__Solve_Chinese_Checkers__VirtualMemory__) */
