//
//  TwoBitMMapVector.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/20/13.
//
//

#include "TwoBitMMapVector.h"
#include <cstdlib>
#include <cstdio>
#include "MMapUtil.h"
#include <sys/mman.h>
#include <strings.h>
#include <string.h>

TwoBitMMapVector::TwoBitMMapVector(uint64_t entries, const char *file, bool zero)
{
	numBits = entries*2;
	numBytes = (numBits+7)/8;
	storage = GetMMAP(file, numBytes, fd, zero); // number of bytes needed
	size = (numBits>>3)+1;
	true_size = entries;
	memmap = true;
}

TwoBitMMapVector::~TwoBitMMapVector()
{
	if (!memmap)
	{
		delete [] storage;
	}
	else {
		// close memmap
		CloseMMap((uint8_t*)storage, numBytes, fd);
	}
}

void TwoBitMMapVector::Advise(int advice)
{
	madvise(storage, true_size/8, advice);
}

void TwoBitMMapVector::Fill(uint8_t value)
{
	memset(storage, value, numBytes);
}

inline uint64_t arrayOffset(uint64_t index)
{ return index>>2; }

inline int byteOffset(uint64_t index)
{ return index&0x3; }

inline uint8_t valueFromByte(uint8_t val, int offset)
{ return (val>>(offset<<1))&3; }

void TwoBitMMapVector::clear()
{
	bzero(storage, numBytes);
}

uint8_t TwoBitMMapVector::Get(uint64_t index) const
{
	if ((arrayOffset(index)) > numBytes)
	{
		printf("GET %llu OUT OF RANGE\n", index);
		exit(0);
	}
	return valueFromByte(storage[arrayOffset(index)], byteOffset(index));
}

void TwoBitMMapVector::Set(uint64_t index, uint8_t value)
{
	if (arrayOffset(index) > size)
	{
		printf("SET %llu OUT OF RANGE\n", index);
		exit(0);
	}
	uint8_t currVal = storage[arrayOffset(index)];
	uint8_t mask = 0xFF^(0x3<<(byteOffset(index)<<1));
	currVal = (currVal&mask)|(value<<(byteOffset(index)<<1));
	storage[arrayOffset(index)] = currVal;
}
