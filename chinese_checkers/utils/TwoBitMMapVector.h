//
//  TwoBitMMapVector.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/20/13.
//
//

#ifndef __Solve_Chinese_Checkers__TwoBitMMapVector__
#define __Solve_Chinese_Checkers__TwoBitMMapVector__

#include <iostream>

#include <stdint.h>
#include "MMapUtil.h"
#include <sys/mman.h>

/**
 * An efficient bit-wise vector implementation.
 */

class TwoBitMMapVector {
public:
	TwoBitMMapVector(uint64_t entries, const char *file, bool zero);
	~TwoBitMMapVector();
	void Advise(int advice);
	void Fill(uint8_t value);
	void clear();
	uint64_t GetSize() { return true_size; }
	uint8_t Get(uint64_t index) const;
	void Set(uint64_t index, uint8_t value);
private:
	uint64_t size, true_size;
	uint64_t numBits, numBytes;
	uint8_t *storage;
	bool memmap;
	int fd;
};

#endif /* defined(__Solve_Chinese_Checkers__TwoBitMMapVector__) */
