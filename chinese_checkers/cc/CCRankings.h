//
//  CCRankings.h
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/25/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#ifndef CCRankings_h
#define CCRankings_h

#include <stdio.h>
#include "CCheckers.h"

constexpr uint64_t ConstFactorial(int n)
{
	return (n < 2)?1:(n*ConstFactorial(n-1));
}

constexpr uint64_t ConstChooseHelper(int n, int k)
{
	return (k > 0)?(n*ConstChooseHelper(n-1, k-1)):1;

}
constexpr uint64_t ConstChoose(int n, int k)
{
	
	return (2*k > n)?(ConstChooseHelper(n, n-k)/ConstFactorial(n-k)):(ConstChooseHelper(n, k)/ConstFactorial(k));
}

constexpr uint64_t ConstGetNonMiddleSymmetry(int diagonalSize, int numOnDiagonal, int numElsewhere)
{
	return (numOnDiagonal < 0)?0:(ConstChoose(diagonalSize, numOnDiagonal)*ConstChoose((diagonalSize-1)*diagonalSize/2, numElsewhere/2)
								  +ConstGetNonMiddleSymmetry(diagonalSize, numOnDiagonal-2, numElsewhere+2));
}

constexpr uint64_t ConstGetSymmetricP1Ranks(int boardDimension, int pieces)
{
	return
	(ConstChoose(boardDimension*boardDimension, pieces)-ConstGetNonMiddleSymmetry(boardDimension, pieces, 0))/2+
	ConstGetNonMiddleSymmetry(boardDimension, pieces, 0);
}

// Ranking interleaves the two players positions in the ranking
class CCDefaultRank {
public:
	int64_t getMaxRank() const { return cc.getMaxRank(); }
	int64_t rank(const CCState &s) const { return cc.rank(s); }
	bool unrank(int64_t r, CCState &s) const { return cc.unrank(r, s); }
	const char *name() { return "default"; }
private:
	CCheckers cc;
};

class CCLocationColorRank  {
public:
	int64_t getMaxRank() const { return cc.getMaxRank(); }
	int64_t rank(const CCState &s) const;
	bool unrank(int64_t r, CCState &s) const;
	const char *name() { return "CCLocColor"; }
private:
	CCheckers cc;
};

class CCColorLocationRank  {
public:
	int64_t getMaxRank() const { return cc.getMaxRank(); }
	int64_t rank(const CCState &s) const;
	bool unrank(int64_t r, CCState &s) const;
	const char *name() { return "CCColorLoc"; }
private:
	CCheckers cc;
};


// Player 1 is ranked and then player 2
class CCLocalRank12 {
public:
	int64_t getMaxRank() const { return cc.getMaxRank(); }
	int64_t rank(const CCState &s) const;
	bool unrank(int64_t r, CCState &s) const;
	const char *name() { return "LOCAL12"; }
private:
	CCheckers cc;
};

class CCLocalRank21 {
public:
	int64_t getMaxRank() const { return cc.getMaxRank(); }
	int64_t rank(const CCState &s) const;
	bool unrank(int64_t r, CCState &s) const;
	const char *name() { return "LOCAL21"; }
private:
	CCheckers cc;
};

// This puts P1 in the high bits and p2 int the low bits.
// P2 is relative to P1
class CCPSRank12 {
public:
	int64_t getMaxRank() const { return cc.getMaxRank()>>1; }
	int64_t rank(const CCState &s) const;
	int64_t rank(const CCState &s, int64_t &p1, int64_t &p2) const;
//	int64_t rankLRSymmetry(const CCState &s, int64_t &p1, int64_t &p2) const;
	int64_t rank(int64_t p1, int64_t p2) const;
	int64_t rankP1(const CCState &s) const;
	int64_t rankP2(const CCState &s) const;
	void GetFirstP1RelP2(CCState &state, int64_t &rank);
	bool IncrementP1RelP2(CCState &state, int64_t &rank);
	bool TryAddR1(CCState &state, int64_t rank);
	bool unrank(int64_t r, CCState &s) const;
	bool unrank(int64_t r1, int64_t r2, CCState &s) const;
	void unrank(int64_t r, int64_t &r1, int64_t &r2) const;
	bool unrankP1(int64_t r1, CCState &s) const;
	bool unrankP2(int64_t r2, CCState &s) const;

	int64_t getMaxP1Rank() const;
	int64_t getMaxP2Rank() const;

	const char *name() { return "PS12"; }
private:
	int GetNextFreeLoc(CCState &s, int start) const;
	CCheckers cc;
};

// This puts P1 in the high bits and p2 int the low bits.
// P1 is relative to P2
class CCPSRank12a {
public:
	int64_t getMaxRank() const { return cc.getMaxRank()>>1; }
	int64_t rank(const CCState &s) const;
	bool unrank(int64_t r, CCState &s) const;
	const char *name() { return "PS12A"; }
private:
	CCheckers cc;
};


// This puts P2 in the high bits and P1 int the low bits.
// P2 is relative to P1
class CCPSRank21 {
public:
	int64_t getMaxRank() const { return cc.getMaxRank()>>1; }
	int64_t rank(const CCState &s) const;
	bool unrank(int64_t r, CCState &s) const;
	const char *name() { return "PS21"; }
private:
	CCheckers cc;
};

// This puts P2 in the high bits and P1 int the low bits.
// P1 is relative to P2
class CCPSRank21a {
public:
	int64_t getMaxRank() const { return cc.getMaxRank()>>1; }
	int64_t rank(const CCState &s) const;
	bool unrank(int64_t r, CCState &s) const;
	const char *name() { return "PS21A"; }
private:
	CCheckers cc;
};


#endif /* CCRankings_h */
