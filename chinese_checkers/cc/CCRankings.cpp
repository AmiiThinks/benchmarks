//
//  CCRankings.cpp
//  CC Solver
//
//  Created by Nathan Sturtevant on 10/25/17.
//  Copyright © 2017 NS Software. All rights reserved.
//

#include "CCRankings.h"
#include <cassert>

int64_t CCLocalRank12::rank(const CCState &s) const
{
	int64_t r0 = cc.rankPlayer(s, 0);
	int64_t r1 = cc.rankPlayerRelative(s, 1, 0);
	assert(r0 < cc.getMaxSinglePlayerRank());
	assert(r1 >= 0);
	assert(r1 < cc.getMaxSinglePlayerRankRelative());
	
	//	return (r1*cc.getMaxSinglePlayerRank()+r0)*2+s.toMove;
	return (r0*cc.getMaxSinglePlayerRankRelative()+r1)*2+s.toMove;
}
bool CCLocalRank12::unrank(int64_t r, CCState &s) const
{
	int64_t toMove = r&1;
	r>>=1;
	//	int64_t r0 = r%cc.getMaxSinglePlayerRank();
	//	int64_t r1 = r/cc.getMaxSinglePlayerRank();
	int64_t r0 = r/cc.getMaxSinglePlayerRankRelative();
	int64_t r1 = r%cc.getMaxSinglePlayerRankRelative();
	cc.unrankPlayer(r0, s, 0);
	cc.unrankPlayerRelative(r1, s, 1, 0);
	s.toMove = toMove;
	return true;
}


int64_t CCLocalRank21::rank(const CCState &s) const
{
	int64_t r0 = cc.rankPlayer(s, 0);
	int64_t r1 = cc.rankPlayerRelative(s, 1, 0);
	assert(r0 < cc.getMaxSinglePlayerRank());
	assert(r1 >= 0);
	assert(r1 < cc.getMaxSinglePlayerRankRelative());
	
	return (r1*cc.getMaxSinglePlayerRank()+r0)*2+s.toMove;
}
bool CCLocalRank21::unrank(int64_t r, CCState &s) const
{
	int64_t toMove = r&1;
	r>>=1;
	int64_t r0 = r%cc.getMaxSinglePlayerRank();
	int64_t r1 = r/cc.getMaxSinglePlayerRank();
	cc.unrankPlayer(r0, s, 0);
	cc.unrankPlayerRelative(r1, s, 1, 0);
	s.toMove = toMove;
	return true;
}



int64_t CCPSRank12::getMaxP1Rank() const
{
	return cc.getMaxSinglePlayerRank();
}
int64_t CCPSRank12::getMaxP2Rank() const
{
	return cc.getMaxSinglePlayerRankRelative();
}

int64_t CCPSRank12::rank(int64_t r0, int64_t r1) const
{
	return (r0*cc.getMaxSinglePlayerRankRelative()+r1);
}

void CCPSRank12::unrank(int64_t r, int64_t &r1, int64_t &r2) const
{
	r1 = r/cc.getMaxSinglePlayerRankRelative();
	r2 = r%cc.getMaxSinglePlayerRankRelative();
}

int64_t CCPSRank12::rank(const CCState &s, int64_t &r0, int64_t &r1) const
{
	switch (s.toMove)
	{
		case 0:
		{
//			cc.rank(s, r0, r1);
			r0 = cc.rankPlayer(s, 0);
			r1 = cc.rankPlayerRelative(s, 1, 0);
//			assert(r0 < cc.getMaxSinglePlayerRank());
//			assert(r1 >= 0);
//			assert(r1 < cc.getMaxSinglePlayerRankRelative());
			
			return (r0*cc.getMaxSinglePlayerRankRelative()+r1);
		}
		case 1:
		{
			CCState tmp = s;
			cc.SymmetryFlipVert(tmp);
			//tmp.Reverse();
			{
				r0 = cc.rankPlayer(tmp, 0);
				r1 = cc.rankPlayerRelative(tmp, 1, 0);
				assert(r0 < cc.getMaxSinglePlayerRank());
				assert(r1 >= 0);
				assert(r1 < cc.getMaxSinglePlayerRankRelative());
				
				return (r0*cc.getMaxSinglePlayerRankRelative()+r1);
			}
		}
	}
	return -1;
}


int64_t CCPSRank12::rank(const CCState &s) const
{
	switch (s.toMove)
	{
		case 0:
		{
			int64_t r0 = cc.rankPlayer(s, 0);
			int64_t r1 = cc.rankPlayerRelative(s, 1, 0);
			assert(r0 < cc.getMaxSinglePlayerRank());
			assert(r1 >= 0);
			assert(r1 < cc.getMaxSinglePlayerRankRelative());
			
			return (r0*cc.getMaxSinglePlayerRankRelative()+r1);
		}
		case 1:
		{
			CCState tmp = s;
			//tmp.Reverse();
			cc.SymmetryFlipVert(tmp);

			{
				int64_t r0 = cc.rankPlayer(tmp, 0);
				int64_t r1 = cc.rankPlayerRelative(tmp, 1, 0);
				assert(r0 < cc.getMaxSinglePlayerRank());
				assert(r1 >= 0);
				assert(r1 < cc.getMaxSinglePlayerRankRelative());
				
				return (r0*cc.getMaxSinglePlayerRankRelative()+r1);
			}
		}
	}
	return -1;
}
int64_t CCPSRank12::rankP1(const CCState &s) const
{
	switch (s.toMove)
	{
		case 0:
		{
			return cc.rankPlayer(s, 0);
		}
		case 1:
		{
			CCState tmp = s;
			//tmp.Reverse();
			cc.SymmetryFlipVert(tmp);
			{
				return cc.rankPlayer(tmp, 0);
			}
		}
	}
	return -1;
}
int64_t CCPSRank12::rankP2(const CCState &s) const
{
	switch (s.toMove)
	{
		case 0:
		{
			return cc.rankPlayerRelative(s, 1, 0);
		}
		case 1:
		{
			CCState tmp = s;
			//tmp.Reverse();
			cc.SymmetryFlipVert(tmp);
			{
				return cc.rankPlayerRelative(tmp, 1, 0);
			}
		}
	}
	return -1;
}

/*
int64_t CCPSRank12::rankLRSymmetry(const CCState &s, int64_t &p1, int64_t &p2) const
{
	CCState tmp;
	cc.FlipPlayer(s, tmp, 0);
	cc.FlipPlayer(s, tmp, 1);
	// This is just to see and make sure it is working.
//	for (int x = 0; x < NUM_SPOTS; x++)
//		tmp.board[x] = 0;
//	for (int x = 0; x < NUM_PLAYERS; x++)
//	{
//		for (int y = 0; y < NUM_PIECES; y++)
//			tmp.board[tmp.pieces[x][y]] = x+1;
//	}
//	printf("Before: "); s.PrintASCII();
//	printf(" After: "); tmp.PrintASCII();
	return rank(tmp, p1, p2);
}
*/

bool CCPSRank12::unrank(int64_t r, CCState &s) const
{
	int64_t r0 = r/cc.getMaxSinglePlayerRankRelative();
	int64_t r1 = r%cc.getMaxSinglePlayerRankRelative();
	cc.unrankPlayer(r0, s, 0);
	cc.unrankPlayerRelative(r1, s, 1, 0);
	s.toMove = 0;
	return true;
}
bool CCPSRank12::unrank(int64_t r1, int64_t r2, CCState &s) const
{
	assert(r1 < cc.getMaxSinglePlayerRank());
	assert(r2 < cc.getMaxSinglePlayerRankRelative());
	cc.unrankPlayer(r1, s, 0);
	cc.unrankPlayerRelative(r2, s, 1, 0);
	s.toMove = 0;
	return true;
}
bool CCPSRank12::unrankP1(int64_t r1, CCState &s) const
{
	cc.unrankPlayer(r1, s, 0);
	s.toMove = 0;
	return true;
}
/*
 * Requires that unrankP1 has just been called on the state
 */
bool CCPSRank12::unrankP2(int64_t r2, CCState &s) const
{
	cc.unrankPlayerRelative(r2, s, 1, 0);
	s.toMove = 0;
	return true;
}

bool CCPSRank12::TryAddR1(CCState &state, int64_t rank)
{
	CCState tmp;
	cc.unrankPlayer(rank, tmp, 0);
	int p1 = 0, p2 = 0;
	while (p1 < NUM_PIECES && p2 < NUM_PIECES)
	{
		if (state.pieces[1][p2] == tmp.pieces[0][p1])
			return false;
		if (state.pieces[1][p2] > tmp.pieces[0][p1])
		{
			p2++;
			continue;
		}
		assert(state.pieces[1][p2] < tmp.pieces[0][p1]);
		p1++;
	}
	// state is valid - merge
	for (int x = 0; x < NUM_PIECES; x++)
	{
		state.pieces[0][x] = tmp.pieces[0][x];
		state.board[tmp.pieces[0][x]] = 1;
	}
	return true;
}

void CCPSRank12::GetFirstP1RelP2(CCState &state, int64_t &rank)
{
	assert(NUM_PLAYERS == 2);
	// Reset p1 pieces - works even if state is not valid for p1
	// (More robust code because we don't expect to call this often)
	for (int x = 0; x < NUM_SPOTS; x++)
	{
		if (state.board[x] == 1)
			state.board[x] = 0;
	}
	
	int nextLoc = 0;
	for (int pieceNum = NUM_PIECES-1; pieceNum >= 0; pieceNum--)
	{
		int next = GetNextFreeLoc(state, nextLoc);
		state.pieces[0][pieceNum] = next;
		state.board[next] = 1;
		nextLoc = next+1;
	}
	rank = rankP1(state);
}

/*
 * Pass in the current rank and the new rank is returned
 */
bool CCPSRank12::IncrementP1RelP2(CCState &state, int64_t &rank)
{
	int nextLoc = -1;
	int pieceNum;
	for (pieceNum = 0; pieceNum < NUM_PIECES; pieceNum++)
	{
		nextLoc = GetNextFreeLoc(state, state.pieces[0][pieceNum]);
		if (nextLoc == NUM_SPOTS)
		{
			if (pieceNum == 0)
				rank += NUM_SPOTS-state.pieces[0][pieceNum]-1;
			else
				rank += cc.binomialSum(NUM_SPOTS-state.pieces[0][pieceNum]-2, pieceNum-1, pieceNum);
		}
		else {
			rank += cc.binomialSum(NUM_SPOTS-state.pieces[0][pieceNum]-2, NUM_SPOTS-nextLoc-1, pieceNum);
		}
		if (nextLoc != NUM_SPOTS)
		{
			break;
		}
	}
	rank++;
	if (nextLoc == NUM_SPOTS)
		return false;
	for (int x = pieceNum; x >= 0; x--)
		state.board[state.pieces[0][x]] = 0;
	// set pieceNum & reset board
	for (; pieceNum >= 0; pieceNum--)
	{
		// need to increment rank if nextLoc skips some pieces!
		state.board[nextLoc] = 1;
		state.pieces[0][pieceNum] = nextLoc;
		// set all remaining pieces
		if (pieceNum != 0)
		{
			int oldLoc = nextLoc;
			nextLoc = GetNextFreeLoc(state, nextLoc+1);
			rank += cc.binomialSum(NUM_SPOTS-oldLoc-2, NUM_SPOTS-nextLoc-1, pieceNum-1);
		}
	}
	return true;
}

int CCPSRank12::GetNextFreeLoc(CCState &s, int start) const
{
	for (int x = start; x < NUM_SPOTS; x++)
		if (s.board[x] == 0)
			return x;
	return NUM_SPOTS;
}

int64_t CCPSRank12a::rank(const CCState &s) const
{
	switch (s.toMove)
	{
		case 0:
		{
			int64_t r0 = cc.rankPlayer(s, 1);
			int64_t r1 = cc.rankPlayerRelative(s, 0, 1);
			assert(r0 < cc.getMaxSinglePlayerRank());
			assert(r1 >= 0);
			assert(r1 < cc.getMaxSinglePlayerRankRelative());
			
			return (r1*cc.getMaxSinglePlayerRank()+r0);
		}
		case 1:
		{
			CCState tmp = s;
			tmp.Reverse();
			
			{
				int64_t r0 = cc.rankPlayer(tmp, 1);
				int64_t r1 = cc.rankPlayerRelative(tmp, 0, 1);
				assert(r0 < cc.getMaxSinglePlayerRank());
				assert(r1 >= 0);
				assert(r1 < cc.getMaxSinglePlayerRankRelative());
				
				return (r1*cc.getMaxSinglePlayerRank()+r0);
			}
			
		}
	}
	return -1;
}
bool CCPSRank12a::unrank(int64_t r, CCState &s) const
{
	int64_t r0 = r%cc.getMaxSinglePlayerRank();
	int64_t r1 = r/cc.getMaxSinglePlayerRank();
	cc.unrankPlayer(r0, s, 1);
	cc.unrankPlayerRelative(r1, s, 0, 1);
	s.toMove = 0;
	return true;
}


int64_t CCPSRank21::rank(const CCState &s) const
{
	switch (s.toMove)
	{
		case 0:
		{
			int64_t r0 = cc.rankPlayer(s, 0);
			int64_t r1 = cc.rankPlayerRelative(s, 1, 0);
			assert(r0 < cc.getMaxSinglePlayerRank());
			assert(r1 >= 0);
			assert(r1 < cc.getMaxSinglePlayerRankRelative());
			
			return (r1*cc.getMaxSinglePlayerRank()+r0);
		}
		case 1:
		{
			CCState tmp = s;
			tmp.Reverse();

			{
				int64_t r0 = cc.rankPlayer(tmp, 0);
				int64_t r1 = cc.rankPlayerRelative(tmp, 1, 0);
				assert(r0 < cc.getMaxSinglePlayerRank());
				assert(r1 >= 0);
				assert(r1 < cc.getMaxSinglePlayerRankRelative());
				
				return (r1*cc.getMaxSinglePlayerRank()+r0);
			}

		}
	}
	return -1;
}
bool CCPSRank21::unrank(int64_t r, CCState &s) const
{
	int64_t r0 = r%cc.getMaxSinglePlayerRank();
	int64_t r1 = r/cc.getMaxSinglePlayerRank();
	cc.unrankPlayer(r0, s, 0);
	cc.unrankPlayerRelative(r1, s, 1, 0);
	s.toMove = 0;
	return true;
}

int64_t CCPSRank21a::rank(const CCState &s) const
{
	switch (s.toMove)
	{
		case 0:
		{
			int64_t r0 = cc.rankPlayer(s, 1);
			int64_t r1 = cc.rankPlayerRelative(s, 0, 1);
			assert(r0 < cc.getMaxSinglePlayerRank());
			assert(r1 >= 0);
			assert(r1 < cc.getMaxSinglePlayerRankRelative());
			
			return (r0*cc.getMaxSinglePlayerRankRelative()+r1);
		}
		case 1:
		{
			CCState tmp = s;
			tmp.Reverse();
			
			{
				int64_t r0 = cc.rankPlayer(tmp, 1);
				int64_t r1 = cc.rankPlayerRelative(tmp, 0, 1);
				assert(r0 < cc.getMaxSinglePlayerRank());
				assert(r1 >= 0);
				assert(r1 < cc.getMaxSinglePlayerRankRelative());
				
				return (r0*cc.getMaxSinglePlayerRankRelative()+r1);
			}
		}
	}
	return -1;
}
bool CCPSRank21a::unrank(int64_t r, CCState &s) const
{
	int64_t r0 = r/cc.getMaxSinglePlayerRankRelative();
	int64_t r1 = r%cc.getMaxSinglePlayerRankRelative();
	cc.unrankPlayer(r0, s, 1);
	cc.unrankPlayerRelative(r1, s, 0, 1);
	s.toMove = 0;
	return true;
}

uint64_t nchoosek(int n, int k)
{
	uint64_t val = 1;
	for (int x = 0; x < k; x++)
		val *= (n-x);
	for (int x = 2; x <= k; x++)
		val /= x;
	return val;
}

uint64_t rank(int *board, int count, int spaces, int offset = 0)
{
	if (count == 0)
		return 0;
	if (board[0]-offset == 0)
		return rank(&board[1], count-1, spaces-1, offset+1);
	uint64_t res = nchoosek(spaces-1, count-1);
	return res+rank(board, count, spaces-1, offset+1);
}

void unrank(uint64_t rank, int *board, int count, int spaces, int total)
{
	if (count == 0)
		return;
	uint64_t res = nchoosek(spaces-1, count-1);
	if (rank >= res)
		unrank(rank-res, board, count, spaces-1, total);
	else {
		board[0] = total-spaces;
		unrank(rank, &board[1], count-1, spaces-1, total);
	}
}


int64_t CCLocationColorRank::rank(const CCState &s) const
{
	int which = 0, which2 = 0;
	int board[2*NUM_PIECES];
	int board2[NUM_PIECES];
	for (int x = 0; x < NUM_SPOTS; x++)
	{
		if (s.board[x])
		{
			board[which] = x;
			which++;
			if (s.board[x] == 1)
			{
				board2[which2] = which;
				which2++;
			}
		}
	}
	int64_t r1 = ::rank(board, 2*NUM_PIECES, NUM_SPOTS);
	int64_t r2 = ::rank(board2, NUM_PIECES, 2*NUM_PIECES);
	int64_t r2_count = nchoosek(2*NUM_PIECES, NUM_PIECES);
	return r1*r2_count+r2;
}

bool CCLocationColorRank::unrank(int64_t r, CCState &s) const
{
	int board[2*NUM_PIECES];
	int board2[NUM_PIECES];
	int which = 0, which2 = 0;
	int p1 = NUM_PIECES-1;
	int p2 = NUM_PIECES-1;
	int64_t r2_count = nchoosek(2*NUM_PIECES, NUM_PIECES);
	int64_t r1 = r/r2_count;
	int64_t r2 = r%r2_count;
	::unrank(r1, board, 2*NUM_PIECES, NUM_SPOTS, NUM_SPOTS);
	::unrank(r2, board2, NUM_PIECES, 2*NUM_PIECES, 2*NUM_PIECES);

	for (int x = 0; x < NUM_SPOTS; x++)
	{
		if (x != board[which])
		{
			s.board[x] = 0;
		}
		else {
			if (board2[which2] == which)
			{
				s.board[x] = 1;
				which2++;
			}
			else {
				s.board[x] = 2;
			}
			which++;
		}
	}
	p1 = 0;
	p2 = 0;
	for (int x = NUM_SPOTS-1; x >= 0; x--)
	{
		if (s.board[x] == 1)
		{
			s.pieces[0][p1] = x;
			p1++;
		}
		else if (board[x] == 2)
		{
			s.pieces[1][p2] = x;
			p2++;
		}
	}
	return true;
}

int64_t CCColorLocationRank::rank(const CCState &s) const
{
	int which = 0, which2 = 0;
	int board[2*NUM_PIECES];
	int board2[NUM_PIECES];
	for (int x = 0; x < NUM_SPOTS; x++)
	{
		if (s.board[x])
		{
			board[which] = x;
			which++;
			if (s.board[x] == 1)
			{
				board2[which2] = which;
				which2++;
			}
		}
	}
	int64_t r1 = ::rank(board, 2*NUM_PIECES, NUM_SPOTS);
	int64_t r2 = ::rank(board2, NUM_PIECES, 2*NUM_PIECES);
	int64_t r1_count = nchoosek(NUM_SPOTS, 2*NUM_PIECES);
	return r2*r1_count+r1;
}

bool CCColorLocationRank::unrank(int64_t r, CCState &s) const
{
	int board[2*NUM_PIECES];
	int board2[NUM_PIECES];
	int which = 0, which2 = 0;
	int p1 = NUM_PIECES-1;
	int p2 = NUM_PIECES-1;
	int64_t r1_count = nchoosek(NUM_SPOTS, 2*NUM_PIECES);
	int64_t r1 = r%r1_count;
	int64_t r2 = r/r1_count;
	::unrank(r1, board, 2*NUM_PIECES, NUM_SPOTS, NUM_SPOTS);
	::unrank(r2, board2, NUM_PIECES, 2*NUM_PIECES, 2*NUM_PIECES);
	
	for (int x = 0; x < NUM_SPOTS; x++)
	{
		if (x != board[which])
		{
			s.board[x] = 0;
		}
		else {
			if (board2[which2] == which)
			{
				s.board[x] = 1;
				which2++;
			}
			else {
				s.board[x] = 2;
			}
			which++;
		}
	}
	p1 = 0;
	p2 = 0;
	for (int x = NUM_SPOTS-1; x >= 0; x--)
	{
		if (s.board[x] == 1)
		{
			s.pieces[0][p1] = x;
			p1++;
		}
		else if (board[x] == 2)
		{
			s.pieces[1][p2] = x;
			p2++;
		}
	}
	return true;
}

