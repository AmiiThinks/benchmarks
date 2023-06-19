//
//  VirtualMemBitVector.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 11/29/13.
//
//

#include "VirtualMemBitVector.h"
#include <string.h>

// structure:
// one set of pointers, each pointer of a fixed size [const]
// bucket of data
// limit of how many buckets we can store


// potential enhancements:
// * put I/O in another thread(?)

VirtualMemBitVector::VirtualMemBitVector(int numEntries_, int maxMemoryUsage_)
{
	numEntries = numEntries_;
	maxMemoryUsage = maxMemoryUsage_;
	arraySize = (numEntries+bytesPerBlock-1)/bytesPerBlock;
	vm = new dataBlockPointer[arraySize];
	pagedToDisk.resize(arraySize);
	for (uint64_t x = 0; x < arraySize; x++)
		vm[x] = 0;
}


VirtualMemBitVector::~VirtualMemBitVector()
{
	delete [] vm;
}

void VirtualMemBitVector::clear()
{
	for (int x = 0; x < arraySize; x++)
	{
		if (vm[x] != 0)
		{
			FreeBlock(vm[x]);
			vm[x] = 0;
		}
	}
}

// TODO: make const again
bool VirtualMemBitVector::Get(uint64_t index) //const
{
	if (vm[index>>bytesBits] == 0)
	{
		if (pagedToDisk[index>>bytesBits] == false)
		{
			return false;
		}
		else {
			// TOOD: unpage from disk (may page other stuff out)
			PageIn(index>>bytesBits);
		}
	}
	uint64_t localOffset = (index&bytesMask)>>storageBitsPower;
	return (vm[index>>bytesBits]->data[localOffset]>>(index&storageMask))&0x1;
//	if ((index>>storageBitsPower) > size) {
//		printf("GET %llu OUT OF RANGE\n", index);
//		exit(0);
//	}
//	return (((storage[index>>storageBitsPower])>>(index&storageMask))&0x1);
}

void VirtualMemBitVector::Set(uint64_t index, bool value)
{
	if (vm[index>>bytesBits] == 0)
	{
		if (pagedToDisk[index>>bytesBits] == false)
		{
			vm[index>>bytesBits] = GetBlock();
		}
		else {
			// TOOD: unpage from disk (may page other stuff out)
			PageIn(index>>bytesBits);
		}
	}
//	if ((index>>storageBitsPower) > size) {
//		printf("SET %llu OUT OF RANGE\n", index);
//		exit(0);
//	}
//	if (value)
//		storage[index>>storageBitsPower] = storage[index>>storageBitsPower]|(1<<(index&storageMask));
//	else
//		storage[index>>storageBitsPower] = storage[index>>storageBitsPower]&(~(1<<(index&storageMask)));
}

VirtualMemBitVector::dataBlock *VirtualMemBitVector::GetBlock()
{
	if (cache.size() > 0)
	{
		dataBlock *tmp = cache.back();
		cache.pop_back();
		memset(tmp, 0, bytesPerBlock*8/storageBits);
		return tmp;
	}
	return new dataBlock;
}

void VirtualMemBitVector::FreeBlock(dataBlock *c)
{
	cache.push_back(c);
}

void VirtualMemBitVector::PageIn(uint64_t index)
{
	
}

