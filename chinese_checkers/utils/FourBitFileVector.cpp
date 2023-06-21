//
//  FourBitFileVector.cpp
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/29/14.
//
//

#include "FourBitFileVector.h"
#include <stdlib.h>
#include <cassert>

FourBitFileVector::FourBitFileVector(uint64_t entries, const char *file, int cache)
:numEntries(entries), arraySize((entries+63)/64), cacheSize(cache), fname(file)
{
	// cache size must be a power of two
	// totally ineffecient, but correct
	while ((cacheSize & (cacheSize - 1)) != 0)
		cacheSize++;
	
	backingFile = fopen(fname.c_str(), "r+");
	if (backingFile  == 0)
	{
		backingFile = fopen(fname.c_str(), "w+");
		if (backingFile == 0)
		{
			printf("Unable to open file: '%s'\n", file);
			exit(0);
		}
		fclose(backingFile);
		backingFile = fopen(fname.c_str(), "r+");
		if (backingFile == 0)
		{
			printf("Unable to open file: '%s'\n", file);
			exit(0);
		}
	}
	
	storageCache = new uint64_t[cacheSize];
	
	numSectors = (entries+16*cacheSize-1)/(16*cacheSize);
	
	currSector = 0;
	fseek(backingFile, currSector*cacheSize*sizeof(uint64_t), SEEK_SET);
	fread(storageCache, sizeof(uint64_t), cacheSize, backingFile);
	dirty = false;
	
}

FourBitFileVector::~FourBitFileVector()
{
	if (dirty)
	{
		fseek(backingFile, currSector*cacheSize*sizeof(uint64_t), SEEK_SET);
		fwrite(storageCache, sizeof(uint64_t), cacheSize, backingFile);
	}
	fclose(backingFile);
	delete [] storageCache;
}

void FourBitFileVector::Fill(uint64_t value)
{
	for (int x = 0; x < numSectors; x++)
	{
		LoadSector(x);
		for (int y = 0; y < cacheSize; y++)
		{
			storageCache[y] = value;
		}
		dirty = true;
	}
}

uint8_t FourBitFileVector::Get(uint64_t index) const
{
	LoadSector(GetSector(index));
//	printf("Index: %llu: Sector: %llu, SecOffset: %llu, BitOffset: %d\n",
//		   index, GetSector(index), GetSectorOffset(index), GetBitOffset(index));
	return GetBits(storageCache[GetSectorOffset(index)], GetBitOffset(index));
}

void FourBitFileVector::Set(uint64_t index, uint8_t value)
{
	LoadSector(GetSector(index));
	SetBits(storageCache[GetSectorOffset(index)], GetBitOffset(index), value);
}

uint64_t FourBitFileVector::GetSector(uint64_t entry) const
{
	return entry/(16*cacheSize);
}

uint64_t FourBitFileVector::GetSectorOffset(uint64_t entry) const
{
	return (entry/16)%cacheSize;
}

int FourBitFileVector::GetBitOffset(uint64_t entry) const
{
	return 4*(entry%16);
}

uint8_t FourBitFileVector::GetBits(uint64_t bits, int offset) const
{
	return (bits>>offset)&0xF;
}

void FourBitFileVector::SetBits(uint64_t &bits, int offset, uint8_t value)
{
	dirty = true;
	// clear bits
	uint64_t mask = 0xF;
	mask <<= offset;
	mask = ~mask;
	bits &= mask;
	bits |= (uint64_t(value)&0xF)<<offset;
}

void FourBitFileVector::LoadSector(uint64_t sector) const
{
	if (currSector == sector)
	{
		return;
	}
	if (sector >= numSectors)
	{
		printf("Error. Accessing sector %llu; %llu total sectors\n", sector, numSectors);
		assert(sector<numSectors);
	}
	if (dirty)
	{
		fseek(backingFile, currSector*cacheSize*sizeof(uint64_t), SEEK_SET);
		fwrite(storageCache, sizeof(uint64_t), cacheSize, backingFile);
	}
	currSector = sector;
	fseek(backingFile, currSector*cacheSize*sizeof(uint64_t), SEEK_SET);
	fread(storageCache, sizeof(uint64_t), cacheSize, backingFile);
}
