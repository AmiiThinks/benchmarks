//
//  VirtualMemBitVector.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 11/29/13.
//
//

#ifndef __Solve_Chinese_Checkers__VirtualMemBitVector__
#define __Solve_Chinese_Checkers__VirtualMemBitVector__

#include <iostream>
#include <vector>

typedef uint8_t storageElement;
const int storageBits = 8;
const int storageBitsPower = 3;
const int storageMask = 0x7;

const int bytesPerBlock = 1024;
const int bytesBits = 10;
const int bytesMask = 0x3FF;

class VirtualMemBitVector {
public:
	VirtualMemBitVector(int numEntries, int maxMemoryUsage);
	~VirtualMemBitVector();
	void clear();
	//int GetSize() { return true_size; }
	bool Get(uint64_t index);// const;
	void Set(uint64_t index, bool value);
private:
	struct dataBlock {
		storageElement data[bytesPerBlock*8/storageBits];
	};
	typedef dataBlock* dataBlockPointer;
	dataBlockPointer *vm;
	std::vector<bool> pagedToDisk;
	std::vector<dataBlock *> cache;
	uint64_t numEntries;
	uint64_t maxMemoryUsage;
	uint64_t arraySize;
	VirtualMemBitVector::dataBlock *GetBlock();
	void FreeBlock(dataBlock *);
	void PageIn(uint64_t index);
};

#endif /* defined(__Solve_Chinese_Checkers__VirtualMemBitVector__) */
