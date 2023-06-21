//
//  Memory.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/16/14.
//
//

#ifndef __Solve_Chinese_Checkers__Memory__
#define __Solve_Chinese_Checkers__Memory__

#include <stdio.h>
#include <stdint.h>

class Memory {
public:
	Memory();
	size_t Alloc(size_t bytes);
	void Free(size_t);
	size_t GetMaxMemory() { return maxUsage; }
	size_t GetCurrMemory() { return currUsage; }
	void Print();
private:
	struct freeList {
		size_t entries;
		bool free;
		freeList *prev;
		freeList *next;
	};
	freeList head;
	size_t currUsage;
	size_t maxUsage;
};

#endif /* defined(__Solve_Chinese_Checkers__Memory__) */
