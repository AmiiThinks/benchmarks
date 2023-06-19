//
//  FourBitFileVector.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/29/14.
//
//

#ifndef __Solve_Chinese_Checkers__FourBitFileVector__
#define __Solve_Chinese_Checkers__FourBitFileVector__

#include <stdio.h>

#include <stdio.h>
#include <stdint.h>
#include <string>

class FourBitFileVector {
public:
	FourBitFileVector(uint64_t entries, const char *file, int cacheSize = 128);
	~FourBitFileVector();
	void Fill(uint64_t value);
	uint8_t Get(uint64_t index) const;
	void Set(uint64_t index, uint8_t value);
private:
	uint64_t GetSector(uint64_t entry) const;
	uint64_t GetSectorOffset(uint64_t entry) const;
	int GetBitOffset(uint64_t entry) const;
	void LoadSector(uint64_t sector) const;
	uint8_t GetBits(uint64_t bits, int offset) const;
	void SetBits(uint64_t &bits, int offset, uint8_t value);
	
	uint64_t arraySize, numEntries;
	uint64_t *storageCache;
	
	int cacheSize;
	uint64_t numSectors;
	std::string fname;
	
	mutable uint64_t currSector;
	mutable FILE *backingFile;
	mutable bool dirty;
};

#endif /* defined(__Solve_Chinese_Checkers__FourBitFileVector__) */
