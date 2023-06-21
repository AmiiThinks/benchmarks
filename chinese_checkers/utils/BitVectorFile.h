//
//  BitVectorFile.h
//  Solve Chinese Checkers
//
//  Created by Nathan Sturtevant on 12/30/13.
//
//

#ifndef __Solve_Chinese_Checkers__BitVectorFile__
#define __Solve_Chinese_Checkers__BitVectorFile__

#include <iostream>

#include <stdint.h>
#include "MMapUtil.h"
#include <sys/mman.h>
#include "BitVector.h"
/**
 * An efficient bit-wise vector implementation.
 */

class BitVectorFile {
public:
	BitVectorFile(const char *file);
	~BitVectorFile();
	uint64_t GetSize() { return true_size; }
	bool Get(uint64_t index) const;
	//void Set(uint64_t index, bool value);
private:
	uint64_t size, true_size;
	mutable uint64_t currOffset, baseOffset;
	FILE *f;
};

#endif /* defined(__Solve_Chinese_Checkers__BitVectorFile__) */
