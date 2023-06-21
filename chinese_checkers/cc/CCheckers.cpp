#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "CCheckers.h"
#include <string.h>
#include <vector>
#include <assert.h>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

//int CCheckers::globalsReset = 0;
//short CCheckers::mapx[NUM_SPOTS];
//short CCheckers::mapy[NUM_SPOTS];
//short CCheckers::reverseMap[COLUMNS][ROWS];
//short CCheckers::distances[NUM_SPOTS][NUM_SPOTS];
//short CCheckers::leftID[NUM_SPOTS];

#ifdef BOARD_121_10 // full board, 10 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
	{111, 112, 113, 114, 115, 116, 117, 118, 119, 120}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 13, 12, 11, 10, 9, 10, 11, 12, 13, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_81_10 // full board, 10 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
	{71, 72, 73, 74, 75, 76, 77, 78, 79, 80}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_81_8 // full board, 8 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5, 7, 8},
	{72, 73, 75, 76, 77, 78, 79, 80}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif


#ifdef BOARD_81_7 // full board, 7 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 5, 7, 8},
   {72, 73, 75, 77, 78, 79, 80}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_81_6 // full board, 6 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5},
	{75, 76, 77, 78, 79, 80}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_81_5 // full board, 5 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 5},
	{75, 77, 78, 79, 80}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_81_4 // full board, 4 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 4},
	{80, 79, 78, 76}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_73_6 // small board, 6 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5},
	{72, 71, 70, 69, 68, 67}};
short g_count[ROWS+1] =
{1, 2, 3, 10, 9, 8, 7, 8, 9, 10, 3, 2, 1, 0};
#endif

#ifdef BOARD_49_6 // small board, 5 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5},
	{48, 47, 46, 45, 44, 43}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_49_5 // small board, 5 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{1, 2, 3, 4, 5},
	{47, 46, 45, 44, 43}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_49_4 // small board, 4 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 4},
	{48, 47, 46, 44}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_49_3 // small board, 3 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2},
	{46, 47, 48}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

//	{72, 71, 70}};
//{1, 2, 3, 10, 9, 8, 7, 8, 9, 10, 3, 2, 1, 0};

#ifdef BOARD_49_2 // small board, 2 pieces, no extra stars
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1},
	{47, 48}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_49_1 // small board, 2 pieces, no extra stars
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0},
	{48}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_36_6 // smaller board, 6 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5},
	{35, 34, 33, 32, 31, 30}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_36_4 // smaller board, 4 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 4},
	{35, 34, 33, 31}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_25_6 // smaller board, 6 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5},
	{24, 23, 22, 21, 20, 19}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 5, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_16_6 // smaller board, 6 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 3, 4, 5},
	{15, 14, 13, 12, 11, 10}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 3, 2, 1, 0};
#endif

#ifdef BOARD_16_4 // smaller board, 6 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2, 4},
	{15, 14, 13, 11}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 3, 2, 1, 0};
#endif


#ifdef BOARD_16_3 // smaller board, 6 pieces
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2},
	{15, 14, 13}};
short g_count[ROWS+1] =
{1, 2, 3, 4, 3, 2, 1, 0};
#endif


#ifdef BOARD_9_3 // really small board, 2 pieces, no extra stars
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0, 1, 2},
	{8, 7, 6}};
short g_count[ROWS+1] =
{1, 2, 3, 2, 1, 0};
#endif


#ifdef BOARD_9_1 // really small board, 2 pieces, no extra stars
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0},
	{8}};
short g_count[ROWS+1] =
{1, 2, 3, 2, 1, 0};
#endif

#ifdef BOARD_4_1 // really small board, 2 pieces, no extra stars
short g_places[NUM_PLAYERS][NUM_PIECES]
= {{0},
	{3}};
short g_count[ROWS+1] =
{1, 2, 1, 0};
#endif



// Adapted from https://stackoverflow.com/questions/2786899/fastest-sort-of-fixed-length-6-int-array
static __inline__ void local_sort(int *d){
#define MIN(x, y) (x<y?x:y)
#define MAX(x, y) (x<y?y:x)
#define SWAP(x,y) { const int a = MAX(d[x], d[y]); const int b = MIN(d[x], d[y]); d[x] = a; d[y] = b; }
	if (NUM_PIECES == 2)
	{
		SWAP(0, 1);
	}
	else if (NUM_PIECES == 3)
	{
		SWAP(1, 2);
		SWAP(0, 2);
		SWAP(0, 1);
	}
	else if (NUM_PIECES == 4)
	{
		SWAP(0, 1);
		SWAP(2, 3);
		SWAP(0, 2);
		SWAP(1, 3);
		SWAP(1, 2);
	}
	else if (NUM_PIECES == 6)
	{
		SWAP(1, 2);
		SWAP(0, 2);
		SWAP(0, 1);
		SWAP(4, 5);
		SWAP(3, 5);
		SWAP(3, 4);
		SWAP(0, 3);
		SWAP(1, 4);
		SWAP(2, 5);
		SWAP(2, 4);
		SWAP(1, 3);
		SWAP(2, 3);
	}
	else if (NUM_PIECES == 10)
	{
		SWAP(0, 1);
		SWAP(3, 4);
		SWAP(2, 4);
		SWAP(2, 3);
		SWAP(0, 3);
		SWAP(0, 2);
		SWAP(1, 4);
		SWAP(1, 3);
		SWAP(1, 2);
		SWAP(5, 6);
		SWAP(8, 9);
		SWAP(7, 9);
		SWAP(7, 8);
		SWAP(5, 8);
		SWAP(5, 7);
		SWAP(6, 9);
		SWAP(6, 8);
		SWAP(6, 7);
		SWAP(0, 5);
		SWAP(1, 6);
		SWAP(1, 5);
		SWAP(2, 7);
		SWAP(3, 8);
		SWAP(4, 9);
		SWAP(4, 8);
		SWAP(3, 7);
		SWAP(4, 7);
		SWAP(2, 5);
		SWAP(3, 6);
		SWAP(4, 6);
		SWAP(3, 5);
		SWAP(4, 5);
	}
	else {
		int i, j;
		for (i = 1; i < NUM_PIECES; i++) {
			int tmp = d[i];
			for (j = i; j >= 1 && tmp > d[j-1]; j--)
				d[j] = d[j-1];
			d[j] = tmp;
		}
	}
#undef SWAP
#undef MIN
#undef MAX
}

// return CCState object given a board position
CCState listToCCState(const std::vector<int>& board_list, int toMove)
{
	CCState state;

	if (board_list.size() != NUM_SPOTS)
	{
		std::cout << "Error: listToCCState: board_list.size() != NUM_SPOTS" << std::endl;
		exit(1);
	}

	// copy board_list to state.board
	for (int x = 0; x < NUM_SPOTS; x++)
	{
		state.board[x] = board_list[x];
	}

	// set pieces
	state.SetPiecesFromBoard();

	// set to_move
	state.toMove = toMove;

	return state;

}

void CCState::SetPiecesFromBoard() {
    // Clear current pieces
    for (int player = 0; player < NUM_PLAYERS; ++player) {
        for (int piece = 0; piece < NUM_PIECES; ++piece) {
            pieces[player][piece] = 0;
        }
    }

    // Update pieces based on board
    for (int i = 0; i < NUM_SPOTS; ++i) {
        int piece = board[i];
        if (piece != 0) {
            int player = piece - 1;
            int piece_idx = 0;
            while (pieces[player][piece_idx] != 0 && piece_idx < NUM_PIECES) {
                ++piece_idx;
            }
            if (piece_idx < NUM_PIECES) {
                pieces[player][piece_idx] = i;
            }
        }
    }

    // Sort the pieces for each player
    for (int player = 0; player < NUM_PLAYERS; ++player) {
        std::sort(pieces[player], pieces[player] + NUM_PIECES, std::greater<int>());
    }
}




//int flip[NUM_SPOTS];
void CCheckers::initFlip()
{
	int middle, x1, y1;
	CCheckers::toxy(0, middle, y1);
	for (int x = 0; x < NUM_SPOTS; x++)
	{
		CCheckers::toxy(x, x1, y1);
		x1 = middle+(middle-x1);
		flip[x] = CCheckers::fromxy(x1, y1);
	}
}

void CCheckers::FlipPlayer(const CCState &src, CCState &dst, int x) const
{
//	for (int y = 0; y < NUM_PIECES; y++)
//	{
//		int next = flip[pieces[x][y]];
//		for (int z = 0; z < y; z++)
//		{
//			if (s.pieces[x][z] < next)
//			{
//				int tmp = s.pieces[x][z];
//				s.pieces[x][z] = next;
//				next = tmp;
//			}
//		}
//		s.pieces[x][y] = next;
//	}
	dst.toMove = src.toMove;
	for (int y = 0; y < NUM_PIECES; y++)
		dst.pieces[x][y] = flip[src.pieces[x][y]];
//	if (src.pieces[x][0] == 12)
//		printf("%d => %d\n", src.pieces[x][0], flip[src.pieces[x][0]]);
	int start = 0;
	for (int y = 1; y < NUM_PIECES; y++)
	{
//		dst.pieces[x][y] = flip[src.pieces[x][y]];
//		printf("%d => %d\n", src.pieces[x][y], flip[src.pieces[x][y]]);
		if ((dst.pieces[x][y] < dst.pieces[x][y-1]))
		{
			for (unsigned int t = 0; t < (y-start)/2; t++)
			{
				int tmp = dst.pieces[x][start+t];
				dst.pieces[x][start+t] = dst.pieces[x][y-t-1];
				dst.pieces[x][y-t-1] = tmp;
			}
			start = y;
		}
	}
	for (unsigned int t = 0; t < (NUM_PIECES-start)/2; t++)
	{
		int tmp = dst.pieces[x][start+t];
		dst.pieces[x][start+t] = dst.pieces[x][NUM_PIECES-t-1];
		dst.pieces[x][NUM_PIECES-t-1] = tmp;
	}
//	for (int y = 1; y < NUM_PIECES; y++)
//		assert(dst.pieces[x][y] < dst.pieces[x][y-1]);
//	std::sort(dst.pieces[x], dst.pieces[x]+NUM_PIECES, std::greater<int>());
}

void CCheckers::SymmetryFlipHoriz(CCState &s) const
{
	memset(s.board, 0, NUM_SPOTS*sizeof(int));
	int middle, x1, y1;
	CCheckers::toxy_fast(0, middle, y1);
	
	for (int x = 0; x < NUM_PLAYERS; x++)
	{
		for (int y = 0; y < NUM_PIECES; y++)
		{
			CCheckers::toxy_fast(s.pieces[x][y], x1, y1);
			x1 = middle+(middle-x1);
			s.pieces[x][y] = CCheckers::fromxy(x1, y1);
			s.board[s.pieces[x][y]] = x+1;
		}
		local_sort(s.pieces[x]);
//		std::sort(s.pieces[x], s.pieces[x]+NUM_PIECES, std::greater<int>());
	}
	//Verify();
}

void CCheckers::SymmetryFlipHoriz(const CCState &s, CCState &result) const
{
	memset(result.board, 0, NUM_SPOTS*sizeof(int));
	int middle, x1, y1;
	CCheckers::toxy_fast(0, middle, y1);
	
	for (int x = 0; x < NUM_PLAYERS; x++)
	{
		for (int y = 0; y < NUM_PIECES; y++)
		{
			CCheckers::toxy_fast(s.pieces[x][y], x1, y1);
			x1 = middle+(middle-x1);
			result.pieces[x][y] = CCheckers::fromxy(x1, y1);
			result.board[result.pieces[x][y]] = x+1;
		}
		local_sort(result.pieces[x]);
		//std::sort(result.pieces[x], result.pieces[x]+NUM_PIECES, std::greater<int>());
	}
	//Verify();
}

void CCheckers::SymmetryFlipHoriz_PO(CCState &s) const
{
	int middle, x1, y1;
	CCheckers::toxy_fast(0, middle, y1);

	for (int x = 0; x < NUM_PLAYERS; x++)
	{
		for (int y = 0; y < NUM_PIECES; y++)
		{
			CCheckers::toxy_fast(s.pieces[x][y], x1, y1);
			x1 = middle+(middle-x1);
			s.pieces[x][y] = CCheckers::fromxy(x1, y1);
		}
		local_sort(s.pieces[x]);
		//std::sort(s.pieces[x], s.pieces[x]+NUM_PIECES, std::greater<int>());
	}
}

void CCheckers::SymmetryFlipHoriz_PO(const CCState &s, CCState &tmp) const
{
	int middle, x1, y1;
	CCheckers::toxy_fast(0, middle, y1);
	
	for (int x = 0; x < NUM_PLAYERS; x++)
	{
		for (int y = 0; y < NUM_PIECES; y++)
		{
			CCheckers::toxy_fast(s.pieces[x][y], x1, y1);
			x1 = middle+(middle-x1);
			tmp.pieces[x][y] = CCheckers::fromxy(x1, y1);
		}
		local_sort(tmp.pieces[x]);
	}
	tmp.toMove = s.toMove;
}

void CCheckers::SymmetryFlipHorizVert(CCState &s) const
{
	assert(NUM_PLAYERS == 2);
	
	// unclear if memset would be faster here
	//	memset(board, 0, NUM_SPOTS*sizeof(int));
	for (int x = 0; x < 2; x++)
		for (int y = 0; y < NUM_PIECES; y++)
			s.board[s.pieces[x][y]] = 0;
	
	for (int y = 0; y < NUM_PIECES; y++)
	{
		int tmp = s.pieces[0][y];
		s.pieces[0][y] = NUM_SPOTS-s.pieces[1][y]-1;
		s.pieces[1][y] = NUM_SPOTS-tmp-1;
		s.board[s.pieces[0][y]] = 1;
		s.board[s.pieces[1][y]] = 2;
	}
	for (int x = 0; x < 2; x++)
	{
		// keep pieces sorted!
		for (int y = 0; y < NUM_PIECES/2; y++)
		{
			int tmp = s.pieces[x][y];
			s.pieces[x][y] = s.pieces[x][NUM_PIECES-y-1];
			s.pieces[x][NUM_PIECES-y-1] = tmp;
		}
	}
	s.toMove = 1-s.toMove;
}

void CCheckers::SymmetryFlipHorizVert(const CCState &s, CCState &result) const
{
	assert(NUM_PLAYERS == 2);
	
	// unclear if memset would be faster here
	memset(result.board, 0, NUM_SPOTS*sizeof(int));
	
	for (int y = 0; y < NUM_PIECES; y++)
	{
		int tmp = s.pieces[0][y];
		result.pieces[0][y] = NUM_SPOTS-s.pieces[1][y]-1;
		result.pieces[1][y] = NUM_SPOTS-tmp-1;
		result.board[result.pieces[0][y]] = 1;
		result.board[result.pieces[1][y]] = 2;
	}
	for (int x = 0; x < 2; x++)
	{
		// keep pieces sorted!
		for (int y = 0; y < NUM_PIECES/2; y++)
		{
			int tmp = result.pieces[x][y];
			result.pieces[x][y] = result.pieces[x][NUM_PIECES-y-1];
			result.pieces[x][NUM_PIECES-y-1] = tmp;
		}
	}
	result.toMove = 1-s.toMove;
}

// Flip state top to bottom (over/under vertical centerline)
void CCheckers::SymmetryFlipVert(CCState &s) const
{
//	printf("BFR: "); s.PrintASCII();
 	memset(s.board, 0, NUM_SPOTS*sizeof(int));
	int x1, y1;

	for (int y = 0; y < NUM_PIECES; y++)
	{
		CCheckers::toxy_fast(s.pieces[0][y], x1, y1);
		y1 = ROWS+1-y1;
		int p0 = CCheckers::fromxy(x1, y1);
		
		CCheckers::toxy_fast(s.pieces[1][y], x1, y1);
		y1 = ROWS+1-y1;
		int p1 = CCheckers::fromxy(x1, y1);
		
		s.pieces[0][y] = p1;
		s.pieces[1][y] = p0;
		s.board[p1] = 1;
		s.board[p0] = 2;
	}
	local_sort(s.pieces[0]);
	local_sort(s.pieces[1]);
//	std::sort(s.pieces[0], s.pieces[0]+NUM_PIECES, std::greater<int>());
//	std::sort(s.pieces[1], s.pieces[1]+NUM_PIECES, std::greater<int>());
	s.toMove = 1-s.toMove;
//	printf("AFR: "); s.PrintASCII();

	//s.Verify();
}

// P1 pieces are flipped and become P2; P2 pieces are ignored
void CCheckers::SymmetryFlipVertP1(CCState &s) const
{
	//	printf("BFR: "); s.PrintASCII();
	memset(s.board, 0, NUM_SPOTS*sizeof(int));
	int x1, y1;
	
	for (int y = 0; y < NUM_PIECES; y++)
	{
		CCheckers::toxy_fast(s.pieces[0][y], x1, y1);
		y1 = ROWS+1-y1;
		int p0 = CCheckers::fromxy(x1, y1);
		
//		CCheckers::toxy_fast(s.pieces[1][y], x1, y1);
//		y1 = ROWS+1-y1;
//		int p1 = CCheckers::fromxy(x1, y1);
		
//		s.pieces[0][y] = p1;
		s.pieces[1][y] = p0;
//		s.board[p1] = 1;
		s.board[p0] = 2;
	}
//	local_sort(s.pieces[0]);
	local_sort(s.pieces[1]);
	//	std::sort(s.pieces[0], s.pieces[0]+NUM_PIECES, std::greater<int>());
	//	std::sort(s.pieces[1], s.pieces[1]+NUM_PIECES, std::greater<int>());
	s.toMove = 1-s.toMove;
	//	printf("AFR: "); s.PrintASCII();
	
	//s.Verify();
}

void CCheckers::SymmetryFlipVert(const CCState &s, CCState &result) const
{
	//	printf("BFR: "); s.PrintASCII();
	memset(result.board, 0, NUM_SPOTS*sizeof(int));
	int x1, y1;
	
	for (int y = 0; y < NUM_PIECES; y++)
	{
		CCheckers::toxy_fast(s.pieces[0][y], x1, y1);
		y1 = ROWS+1-y1;
		int p0 = CCheckers::fromxy(x1, y1);
		
		CCheckers::toxy_fast(s.pieces[1][y], x1, y1);
		y1 = ROWS+1-y1;
		int p1 = CCheckers::fromxy(x1, y1);
		
		result.pieces[0][y] = p1;
		result.pieces[1][y] = p0;
		result.board[p1] = 1;
		result.board[p0] = 2;
	}
	local_sort(result.pieces[0]);
	local_sort(result.pieces[1]);
//	std::sort(result.pieces[0], result.pieces[0]+NUM_PIECES, std::greater<int>());
//	std::sort(result.pieces[1], result.pieces[1]+NUM_PIECES, std::greater<int>());
	result.toMove = 1-s.toMove;
	//	printf("AFR: "); s.PrintASCII();
	
	//s.Verify();
}

void CCState::Reverse()
{
	assert(NUM_PLAYERS == 2);

	// unclear if memset would be faster here
	//	memset(board, 0, NUM_SPOTS*sizeof(int));
	for (int x = 0; x < 2; x++)
		for (int y = 0; y < NUM_PIECES; y++)
			board[pieces[x][y]] = 0;
	
	for (int y = 0; y < NUM_PIECES; y++)
	{
		int tmp = pieces[0][y];
		pieces[0][y] = NUM_SPOTS-pieces[1][y]-1;
		pieces[1][y] = NUM_SPOTS-tmp-1;
		board[pieces[0][y]] = 1;
		board[pieces[1][y]] = 2;
	}
	for (int x = 0; x < 2; x++)
	{
		// keep pieces sorted!
		for (int y = 0; y < NUM_PIECES/2; y++)
		{
			int tmp = pieces[x][y];
			pieces[x][y] = pieces[x][NUM_PIECES-y-1];
			pieces[x][NUM_PIECES-y-1] = tmp;
		}
	}
	toMove = 1-toMove;
//	Verify();
}

void CCState::Verify()
{
	for (int x = 0; x < NUM_PLAYERS; x++)
	{
		for (int y = 0; y < NUM_PIECES; y++)
		{
			assert(board[pieces[x][y]] == x+1);
			if (y > 0)
				assert(pieces[x][y-1] > pieces[x][y]);
		}
	}
	int cnt = 0;
	for (int x = 0; x < NUM_SPOTS; x++)
		if (board[x] == 1)
			cnt++;
	assert(cnt == NUM_PIECES);
}

void CCState::PrintASCII() const
{
	printf("[%d] ", toMove);
	for (int x = 0; x < NUM_SPOTS; x++)
		printf("%d ", board[x]);
	printf("\n");
}

void CCState::Print() const
{
	int x=0, y=0;
	
	//topscr();
	clrscr();
	setcolor(0, 0);

//	for (int z = 0; z < ROWS; z++)
//    {
//		gotoxy(4, z+1);
//		printf("%2d", z+1);
//    }
//	for (int z = 1; z < COLUMNS+1; z++)
//    {
//		gotoxy(z+10, ROWS+2);
//		if (z >= 10)
//			printf("%d", z/10);
//		gotoxy(z+10, ROWS+3);
//		printf("%d", z%10);
//    }
	// TODO: expensive!
	CCheckers cc;
	for (int z = 0; z < NUM_SPOTS; z++)
    {
		cc.toxy(z, x, y);
		gotoxy(x+10, y);
		if (board[z])
		{
			if (toMove == board[z]-1)
				setcolor(30+board[z], 4);
			else
				setcolor(30+board[z], 0);
			printf("%c", '0'+board[z]);
		}
		else {
			setcolor(0, 1);
			setcolor(1, 1);
			//printf("%c", 183);
			printf(".");
		}
    }
	setcolor(0, 0);
	gotoxy(1, ROWS+5);
	printf("\n");
}

void CCheckers::applyState(std::string configuration, CCState &state)
{
//    state.PrintASCII();
//    Reset(state);
    for (int i = 0; i < NUM_SPOTS; i++)
    {
//        printf("%d", configuration[i] - 48);
        state.board[i] = (int) configuration[i] - 48;
    }

    for (int x = 0; x < NUM_PLAYERS; x++)
    {
        int i = 0;
        for (int y = 0; y < NUM_SPOTS; y++)
        {
            if (state.board[y] == x + 1)
            {
                state.pieces[x][i] = y;
                i++;
            }
        }
        local_sort(state.pieces[x]);
    }
    state.toMove = 0;
//    state.Verify();
//    state.PrintASCII();
}

//void initBinomialSums();

CCheckers::CCheckers()
:forbidden(NUM_SPOTS)
{
	globalsReset = 0;
	moveCache = 0;
	for (int x = 0; x < CACHE_SIZE; x++)
	{
		realCache[x].next = moveCache;
		moveCache = &realCache[x];
	}
	ResetGlobals();
	initBinomial();
	initBinomialSums();
	initFlip();
//	moveCache = 0;
}

CCheckers::~CCheckers()
{ 
//	while (moveCache)
//	{
//		CCMove *c = moveCache;
//		moveCache = moveCache->next;
//		delete c;
//	}
}

int CCheckers::GetMoveCacheSize()
{
	if (moveCache != 0)
	{
		int cnt = 0;
		for (CCMove *m = moveCache; m; m = m->next)
			cnt++;
		return cnt;
	}
	return 0;
}

CCMove *CCheckers::getNewMove() const
{
//	if (moveCache)
//	{
//		CCMove *ret = moveCache;
//		moveCache = moveCache->next;
//		ret->next = 0;
//		return ret;
//	}
//	assert(!"Need to increase move cache size!");
	return new CCMove(0);
}

void CCheckers::freeMove(CCMove *m) const
{
	if (m == 0)
		return;
//	CCMove *tmp;
//	do {
//		tmp = m->next;
//		m->next = moveCache;
//		moveCache = m;
//		m = tmp;
//	} while (tmp != 0);
}

void CCheckers::delMove(CCMove *m)
{
    delete m;
}

//int64_t bi(unsigned int n, unsigned int k);
////static std::vector<int64_t> binomials;
//int64_t binomial(unsigned int n, unsigned int k);
//static std::vector<int64_t> theSums;

void CCheckers::initBinomialSums()
{
	if (theSums.size() == 0)
	{
		theSums.resize((NUM_PIECES+1)*(NUM_SPOTS+1));
		//		sums.resize(NUM_PIECES+1);
		for (int x = 0; x <= NUM_PIECES; x++)
		{
			//			sums[x].resize(NUM_SPOTS+1);
			int64_t result = 0;
			for (int y = 0; y <= NUM_SPOTS; y++)
			{
				result+=binomial(y, x);
				//				sums[x][y] = result;
				theSums[x*(NUM_SPOTS+1)+y] = result;
			}
		}
	}
	if (NUM_PIECES > 1)
	{
		rankOffsets.resize(getMaxSinglePlayerRank2());
		int64_t offset = 0;
		for (int64_t x = 0; x < getMaxSinglePlayerRank2(); x++)
		{
			rankOffsets[x] = offset;
			offset += getMaxSinglePlayerRank2(x);
		}
	}
}

int64_t CCheckers::binomialSum(unsigned int n1, unsigned int n2, unsigned int k) const
{
//	static std::vector<std::vector<int64_t> > sums;
	//assert(theSums[k*(NUM_SPOTS+1)+n1]-theSums[k*(NUM_SPOTS+1)+n2] == sums[k][n1]-sums[k][n2]);
	return theSums[k*(NUM_SPOTS+1)+n1]-theSums[k*(NUM_SPOTS+1)+n2];
	//return sums[k][n1]-sums[k][n2];
	//return result;
}

int64_t CCheckers::biSum(unsigned int n1, unsigned int n2, unsigned int k)
{
	int64_t result = 0;
	for (unsigned int x = n1; x > n2; x--)
	{
		result += bi(x, k);
	}
	return result;
}

void CCheckers::initBinomial()
{
	if (binomials.size() == 0)
	{
		for (int x = 0; x <= NUM_SPOTS; x++)
		{
			for (int y = 0; y <= NUM_PLAYERS*NUM_PIECES; y++)
			{
				binomials.push_back(bi(x, y));
			}
		}
	}
}

int64_t CCheckers::binomial(unsigned int n, unsigned int k) const
{
	//assert(bi(n, k) == binomials[n*(1+NUM_PLAYERS*NUM_PIECES)+k]);
	return binomials[n*(1+NUM_PLAYERS*NUM_PIECES)+k];
//	printf("binomial: %d %d\n", n, k);
//	//assert(n >= (k1 + k2));
//	int64_t num = 1;
//	const unsigned int bound = (n - k);
//	while(n > bound)
//	{
//		num *= n--;
//	}
//	
//	int64_t den = 1;
//	while(k > 1)
//	{
//		den *= k--;
//	}
//	return num / den;
}

int64_t CCheckers::bi(unsigned int n, unsigned int k) const
{
	int64_t num = 1;
	const unsigned int bound = (n - k);
	while(n > bound)
	{
		num *= n--;
	}
	
	int64_t den = 1;
	while(k > 1)
	{
		den *= k--;
	}
	return num / den;
}

int64_t CCheckers::multinomial(unsigned int n, unsigned int k1, unsigned int k2) const
{
	//assert(n >= (k1 + k2));
	int64_t num = 1;
	const unsigned int bound = (n - (k1 + k2));
	while(n > bound)
	{
		num *= n--;
	}
	
	static uint64_t table[21] =
	{ 1ll, 1ll, 2ll, 6ll, 24ll, 120ll, 720ll, 5040ll, 40320ll, 362880ll, 3628800ll, 39916800ll, 479001600ll,
		6227020800ll, 87178291200ll, 1307674368000ll, 20922789888000ll, 355687428096000ll,
		6402373705728000ll, 121645100408832000ll, 2432902008176640000ll };

	int64_t den = table[k1]*table[k2];
//	while(k1 > 1)
//	{
//		den *= k1--;
//	}
//	while(k2 > 1)
//	{
//		den *= k2--;
//	}
	return num / den;
}

int64_t CCheckers::getMaxRank() const
{
  // factor of 2 to store next player
  return 2*multinomial(NUM_SPOTS, NUM_PIECES, NUM_PIECES);
}

/*
 * Rank algorithm adapted from "GPU Exploration of Two-Player Games with
 * Perfect Hash Functions" by Stefan Edelkamp, Damian Sulewski,
 * Cengizhan Yücel, Third Annual Symposium on Combinatorial Search, 2010.
 */
int64_t CCheckers::rank(const CCState &s) const
{
	int64_t r = 0;
	unsigned int l1s = NUM_PIECES, l2s = NUM_PIECES;
	for (int i = 0; (l1s + l2s) > 0; ++i)
	{
		//assert(board[i] >= 0 && board[i] <= 2);  // check the colors
		// cout << "DEBUG: board[" << i << "] = " << board[i] << ", l1s = " << l1s << ", l2s = " << l2s << ", r = " << r << endl;
		switch (s.board[i])
		{
			case 2:
				l2s -= 1;
				break;
			case 1:
				if (l2s > 0)
				{
					r = r + multinomial(NUM_SPOTS - i - 1, l1s, l2s-1);
				}
				l1s -= 1;
				break;
			default: // case 0:
				if (l2s > 0)
				{
					r = r + multinomial(NUM_SPOTS - i - 1, l1s, l2s-1);
				}
				if (l1s > 0)
				{
					r = r + multinomial(NUM_SPOTS - i - 1, l1s-1, l2s);
				}
				break;
		}
	}
	// LSB stores the player to move
	return (r << 1) + s.toMove;
}

/*
 * Rank algorithm adapted from "GPU Exploration of Two-Player Games with
 * Perfect Hash Functions" by Stefan Edelkamp, Damian Sulewski,
 * Cengizhan Yücel, Third Annual Symposium on Combinatorial Search, 2010.
 */
bool CCheckers::unrank(int64_t theRank, CCState &s) const
{
        // clear to zero, this is necessary as we may break out of the
        // loop before i gets all the way to NUM_SPOTS-1 (i.e., when
        // all the pieces have been placed)
 	memset(s.board, 0, NUM_SPOTS*sizeof(int));	
  
	// LSB stores player to move.
	s.toMove = theRank & 0x1;
	theRank = theRank >> 1;
	
	unsigned int l1s = NUM_PIECES, l2s = NUM_PIECES;
	for (int i=0; (l1s + l2s) > 0; ++i)
	{
		int64_t value1, value2;
		if (l2s > 0)
		{
			value2 = multinomial(NUM_SPOTS - i - 1, l1s, l2s - 1);
		}
		else {
			value2 = 0;
		}
		if(l1s > 0)
		{
			value1 = multinomial(NUM_SPOTS - i - 1, l1s - 1, l2s);
		}
		else {
			value1 = 0;
		}
		// this block of code guarantees that the element at the ith index gets either 2, 1, 0
		if (theRank < value2)
		{
			if(l2s <= 0){ // trying to place too many 2s
				return false;
			}
			s.board[i] = 2;
			s.pieces[1][l2s-1] = i;
			l2s = l2s - 1;
		}
		else if (theRank < (value1 + value2))
		{
			if (l1s <= 0)
			{ // trying to place too many 1s;
				return false;
			}
			s.board[i] = 1;
			s.pieces[0][l1s-1] = i;
			theRank = theRank - value2;
			l1s = l1s - 1;
		}
		else {	
			s.board[i] = 0;
			theRank = theRank - (value1 + value2);
		}
	}
	return true;
}

int64_t CCheckers::getMaxSinglePlayerRank(int numPieces) const
{
//	return binomial(NUM_SPOTS-(NUM_PIECES-numPieces), numPieces);
	return binomial(NUM_SPOTS, numPieces);
}

int64_t CCheckers::getMaxSinglePlayerRank() const
{
	return binomial(NUM_SPOTS, NUM_PIECES);
}

int64_t CCheckers::getMaxSinglePlayerRank2() const
{
	return binomial(NUM_SPOTS-(NUM_PIECES-2), 2);
}

int64_t CCheckers::getMaxSinglePlayerRank2(int64_t firstIndex) const
{
	unsigned int ls = 2;
	int i = 0;
	for (; ls > 0; ++i)
	{
		int64_t value;
		if (ls > 0)
		{
			value = binomial(NUM_SPOTS-(NUM_PIECES-2) - i - 1, ls - 1);
		}
		else {
			value = 0;
		}
		if (firstIndex < value)
		{
			ls--;
		}
		else {	
			firstIndex -= value;
		}
	}
	return binomial(NUM_SPOTS-i,NUM_PIECES-2);
}

int64_t CCheckers::rankPlayer(const CCState &s, int who) const
{
	int64_t r2 = 0;
	int last = NUM_SPOTS-1;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1, NUM_PIECES-1-x);
		r2 += tmp;
		last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1-1;
	}
	return r2;
}

int64_t CCheckers::rankPlayerFlipped(const CCState &s, int who) const
{
	int64_t r2 = 0;
	int last = NUM_SPOTS-1;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, s.pieces[who][x], NUM_PIECES-1-x);
		r2 += tmp;
		last = s.pieces[who][x]-1;
	}
	return r2;
}

int64_t CCheckers::getMaxSinglePlayerRankRelative() const
{
	return binomial(NUM_SPOTS-NUM_PIECES, NUM_PIECES);
}

int64_t CCheckers::rankPlayerRelative(const CCState &s, int who, int relative) const
{
	int mod[NUM_PIECES];
	int relPos = 0;
	int myPos = 0;
	int offset = 0;
	do {
		// if their piece is smaller (bigger?) than ours, increase their location and our offset
		if (relPos < NUM_PIECES && s.pieces[relative][NUM_PIECES-1-relPos] < s.pieces[who][NUM_PIECES-1-myPos])
		{
			relPos++;
			offset++;
		}
		else {
			// otherwise put our piece into array
			mod[NUM_PIECES-1-myPos] = s.pieces[who][NUM_PIECES-1-myPos]-offset;
			myPos++;
		}
	} while (myPos < NUM_PIECES);

	int64_t r2 = 0;
	int last = NUM_SPOTS-1-NUM_PIECES;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, NUM_SPOTS-mod[NUM_PIECES-1-x]-1-NUM_PIECES, NUM_PIECES-1-x);
		r2 += tmp;
		last = NUM_SPOTS-mod[NUM_PIECES-1-x]-1-NUM_PIECES-1;
	}
	assert(r2 >= 0);
	assert(r2 < getMaxSinglePlayerRankRelative());
	return r2;
}

void CCheckers::rank(const CCState &s, int64_t &r1, int64_t &r2) const
{
	r1 = 0;
	int last = NUM_SPOTS-1;
	
	assert(NUM_PLAYERS == 2);
	int mod[NUM_PIECES];
	int relPos = 0;
	int myPos = 0;
	int offset = 0;
	do {
		// if their piece is smaller (bigger?) than ours, increase their location and our offset
		if (relPos < NUM_PIECES && s.pieces[0][NUM_PIECES-1-relPos] < s.pieces[1][NUM_PIECES-1-myPos])
		{
			r1 += binomialSum(last, NUM_SPOTS-s.pieces[0][NUM_PIECES-1-relPos]-1, NUM_PIECES-1-relPos);
			last = NUM_SPOTS-s.pieces[0][NUM_PIECES-1-relPos]-1-1;

			relPos++;
			offset++;
		}
		else {
			// otherwise put our piece into array
			mod[NUM_PIECES-1-myPos] = s.pieces[1][NUM_PIECES-1-myPos]-offset;
			myPos++;
		}
	} while (myPos < NUM_PIECES);

	for (int x = relPos; x < NUM_PIECES; x++)
	{
		r1 += binomialSum(last, NUM_SPOTS-s.pieces[0][NUM_PIECES-1-x]-1, NUM_PIECES-1-x);
		last = NUM_SPOTS-s.pieces[0][NUM_PIECES-1-x]-1-1;
	}

	
	{
		r2 = 0;
		int last = NUM_SPOTS-1-NUM_PIECES;
		for (int x = 0; x < NUM_PIECES; x++)
		{
			int64_t tmp = binomialSum(last, NUM_SPOTS-mod[NUM_PIECES-1-x]-1-NUM_PIECES, NUM_PIECES-1-x);
			r2 += tmp;
			last = NUM_SPOTS-mod[NUM_PIECES-1-x]-1-NUM_PIECES-1;
		}
	}
	assert(r2 >= 0);
	assert(r2 < getMaxSinglePlayerRankRelative());
}

void CCheckers::unrankPlayerRelativeHelper(int64_t theRank, CCState &s, int who) const
{
	unsigned int ls = NUM_PIECES;
	for (int i=0; ls > 0; ++i)
	{
		int64_t value;
		if (ls > 0)
		{
			value = binomial(NUM_SPOTS - i - 1 - NUM_PIECES, ls - 1);
		}
		else {
			value = 0;
		}
		if (theRank < value)
		{
			s.pieces[who][ls-1] = i;
			ls--;
		}
		else {
			theRank -= value;
		}
	}
	for (int x = 1; x < NUM_PIECES; x++)
		assert(s.pieces[who][x-1] > s.pieces[who][x]);
}

bool CCheckers::unrankPlayerRelative(int64_t r, CCState &s, int who, int relative) const
{
	// Puts pieces in relative location inside s
	unrankPlayerRelativeHelper(r, s, who);

	// Put into absolute location
	int relPos = 0;
	int myPos = 0;
	int offset = NUM_PIECES;

	do {
		if ((relPos < NUM_PIECES) && s.pieces[relative][relPos] >= s.pieces[who][myPos]+offset)
		{
			relPos++;
			offset--;
		}
		else {
			s.pieces[who][myPos] += offset;
			myPos++;
		}
	} while (myPos < NUM_PIECES);

	// Fill values into array
	for (int x = 0; x < NUM_PIECES; x++)
	{
		s.board[s.pieces[who][x]] = 1+who;
	}
	
	return true;
}



void CCheckers::rankPlayer(const CCState &s, int who, int64_t &index1, int64_t &index2) const
{
	index1 = 0;
	int tot = NUM_SPOTS-1-(NUM_PIECES-2);
	int last = tot;
	for (int x = 0; x < 2; x++)
	{
		int64_t tmp = binomialSum(last, tot-s.pieces[who][NUM_PIECES-1-x], (2)-1-x);
		index1 += tmp;
		last = tot-s.pieces[who][NUM_PIECES-1-x]-1;
	}

	index2 = 0;
	last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1-1]-1-1;
	for (int x = 2; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1, NUM_PIECES-1-x);
		index2 += tmp;
		last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1-1;
	}
}

void CCheckers::rankPlayerFlipped(const CCState &s, int who, int64_t &index1, int64_t &index2) const
{
	index1 = 0;
	int tot = NUM_SPOTS-1-(NUM_PIECES-2);
	int last = tot;
	for (int x = 0; x < 2; x++)
	{
		int64_t tmp = binomialSum(last, tot-(NUM_SPOTS-1-s.pieces[who][x]), (2)-1-x);
		index1 += tmp;
		last = tot-(NUM_SPOTS-1-s.pieces[who][x])-1;
	}
	
	index2 = 0;
	last = NUM_SPOTS-(NUM_SPOTS-1-s.pieces[who][1])-1-1;
	for (int x = 2; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, NUM_SPOTS-(NUM_SPOTS-1-s.pieces[who][x])-1, NUM_PIECES-1-x);
		index2 += tmp;
		last = s.pieces[who][x]-1;
	}
}


bool CCheckers::twoPieceNeighbors(int64_t i1, int64_t i2)
{
	CCState s1, s2, s3;
	int64_t i3;
	unrankPlayerFirstTwo(i1, s1, 0);
	unrankPlayerFirstTwo(i2, s2, 0);

	// if a piece is shared, true
	if (s1.pieces[0][0] == s2.pieces[0][0] &&
		s1.pieces[0][1] == s2.pieces[0][1])
	{
		return true;
	}
	// if no piece is shared, false
	if (s1.pieces[0][0] != s2.pieces[0][0] &&
		s1.pieces[0][0] != s2.pieces[0][1] &&
		s1.pieces[0][1] != s2.pieces[0][0] &&
		s1.pieces[0][1] != s2.pieces[0][1])
	{
		return false;
	}
	
	// fill up the rest of the board, varying the 3rd piece location
	// (high at top, low at bottom)
//	unrankPlayerFirstTwo(i1, s3, 0);
//	// 0->NUM_PIECES-1
//	// 1->NUM_PIECES-2
//	s3.pieces[0][NUM_PIECES-1] = s3.pieces[0][1];
//	s3.pieces[0][NUM_PIECES-2] = s3.pieces[0][0];
//	for (int y = 0; y < NUM_PIECES-3; y++)
//	{
//		s3.pieces[0][y] = NUM_SPOTS-y-1;
//		s3.board[NUM_SPOTS-y-1] = 1;
//	}

	// location to put 3rd piece
	for (int x = NUM_SPOTS-NUM_PIECES+2; x > s1.pieces[0][0]; x--)
	{
		// fill up the rest of the board, varying the 3rd piece location
		// (high at top, low at bottom)
		unrankPlayerFirstTwo(i1, s3, 0);
		// 0->NUM_PIECES-1
		// 1->NUM_PIECES-2
		s3.pieces[0][NUM_PIECES-1] = s3.pieces[0][1];
		s3.pieces[0][NUM_PIECES-2] = s3.pieces[0][0];
		for (int y = 0; y < NUM_PIECES-3; y++)
		{
			s3.pieces[0][y] = NUM_SPOTS-y-1;
			s3.board[NUM_SPOTS-y-1] = 1;
		}
		s3.pieces[0][NUM_PIECES-3] = x;
		s3.board[x] = 1;

//		int x1, y1, x2, y2;
//		toxy(s1.pieces[0][0], x1, y1);
//		toxy(x, x2, y2);
//		if (abs(x1-x2) > 2 || abs(y1-y2) > 2)
//			continue;

//		s3.PrintASCII();
		// get all moves
//		for (int piece = 0; piece < 3; piece++)
		{
			CCMove *m = getMoves(s3);
			//CCMove *m = getMovesForPiece(s3, NUM_PIECES-piece-1);
			for (CCMove *t = m; t; t = t->next)
			{
//				if (t->from != s3.pieces[0][NUM_PIECES-1] &&
//					t->from != s3.pieces[0][NUM_PIECES-2] &&
//					t->from != s3.pieces[0][NUM_PIECES-3])
//					continue;
				ApplyMove(s3, t);
				rankPlayerFirstTwo(s3, 0, i3);
				if (i3 == i2)
				{
	//				printf("These two states are neighbors\n");
	//				s1.Print();
	//				s2.Print();
	//				s3.Print();
	//				UndoMove(s3, t);
	//				s3.Print();
					freeMove(m);
					return true;
				}
				UndoMove(s3, t);
			}
			freeMove(m);
		}
//		s3.board[x] = 0;
	}
	
	return false;
}

void CCheckers::rankPlayerFirstTwo(CCState &s, int who, int64_t &index1) const
{
	index1 = 0;
	int tot = NUM_SPOTS-1-(NUM_PIECES-2);
	int last = tot;
	for (int x = 0; x < 2; x++)
	{
		int64_t tmp = binomialSum(last, tot-s.pieces[who][NUM_PIECES-1-x], (2)-1-x);
		index1 += tmp;
		last = tot-s.pieces[who][NUM_PIECES-1-x]-1;
	}
}

void CCheckers::rankPlayerRemaining(CCState &s, int who, int64_t &index2) const
{
	int last;
	index2 = 0;
	last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1-1]-1-1;
	for (int x = 2; x < NUM_PIECES; x++)
	{
		int64_t tmp = binomialSum(last, NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1, NUM_PIECES-1-x);
		index2 += tmp;
		last = NUM_SPOTS-s.pieces[who][NUM_PIECES-1-x]-1-1;
	}
}

void CCheckers::rankPlayerFirstTwo(CCState &s, int who, int64_t &index1, int numPieces)
{
	index1 = 0;
	int tot = NUM_SPOTS-1-(numPieces-2);
	int last = tot;
	for (int x = 0; x < 2; x++)
	{
		int64_t tmp = biSum(last, tot-s.pieces[who][numPieces-1-x], (2)-1-x);
		index1 += tmp;
		last = tot-s.pieces[who][numPieces-1-x]-1;
	}
}

bool CCheckers::validInFullBoard(const CCState &s, int numPieces)
{
	return (s.pieces[0][numPieces-2] <= NUM_SPOTS-(NUM_PIECES-2)-1);
}

void CCheckers::rankPlayerFirstTwoInFullBoard(CCState &s, int who, int64_t &index1, int numPieces)
{
	assert(validInFullBoard(s, numPieces));
	index1 = 0;
	int tot = NUM_SPOTS-1-(NUM_PIECES-2);
	int last = tot;
	for (int x = 0; x < 2; x++)
	{
		int64_t tmp = biSum(last, tot-s.pieces[who][numPieces-1-x], (2)-1-x);
		index1 += tmp;
		last = tot-s.pieces[who][numPieces-1-x]-1;
	}
}


bool CCheckers::unrankPlayerFirstTwo(int64_t theRank, CCState &s, int who)
{
	int tag = who + 1;
	unsigned int ls = 2; // 2 pieces
	memset(s.board, 0, NUM_SPOTS*sizeof(int));
	for (int i=0; ls > 0; ++i)
	{
		int64_t value;
		if (ls > 0)
		{
			value = binomial(NUM_SPOTS-(NUM_PIECES-2) - i - 1, ls - 1);
		}
		else {
			value = 0;
		}
		if (theRank < value)
		{
			s.board[i] = tag;
			s.pieces[who][ls-1] = i;
			ls--;
		}
		else {
			s.board[i] = 0;
			theRank -= value;
		}
	}
	for (int x = 1; x < 2; x++)
		assert(s.pieces[who][x-1] > s.pieces[who][x]);
	s.toMove = who;
	return true;
}

bool CCheckers::unrankPlayer(int64_t rank1, int64_t rank2, CCState &s, int who)
{
	int64_t realRank = rankOffsets[rank1]+rank2;//rank1*getMaxSinglePlayerRank2(rank1)+rank2;
	return unrankPlayer(realRank, s, who);
}

// returns true if it is a valid unranking given existing pieces
bool CCheckers::unrankPlayer(int64_t theRank, CCState &s, int who) const
{
	int tag = who + 1;
	unsigned int ls = NUM_PIECES;
	memset(s.board, 0, NUM_SPOTS*sizeof(int));	
	for (int i=0; ls > 0; ++i)
	{
		int64_t value;
		if (ls > 0)
		{
			value = binomial(NUM_SPOTS - i - 1, ls - 1);
		}
		else {
			value = 0;
		}
		if (theRank < value)
		{
			s.board[i] = tag;
			s.pieces[who][ls-1] = i;
			ls--;
		}
		else {	
			s.board[i] = 0;
			theRank -= value;
		}
	}
	for (int x = 1; x < NUM_PIECES; x++)
		assert(s.pieces[who][x-1] > s.pieces[who][x]);
	s.toMove = who;
	return true;
}

bool CCheckers::unrankPlayer(int64_t theRank, CCState &s, int who, int numPieces) const
{
	int tag = who + 1;
	unsigned int ls = numPieces;
	memset(s.board, 0, NUM_SPOTS*sizeof(int));
	for (int i=0; ls > 0; ++i)
	{
		int64_t value;
		if (ls > 0)
		{
			// use bi instead of binomial which isn't cached here
			// since we have a different number of pieces
			value = bi(NUM_SPOTS - i - 1, ls - 1);
			//value = bi((NUM_SPOTS-(NUM_PIECES-numPieces)) - i - 1, ls - 1);
		}
		else {
			value = 0;
		}
		if (theRank < value)
		{
			s.board[i] = tag;
			s.pieces[who][ls-1] = i;
			ls--;
		}
		else {
			s.board[i] = 0;
			theRank -= value;
		}
	}
	for (int x = numPieces; x < NUM_PIECES; x++)
		s.pieces[who][x] = 0;
	for (int x = 1; x < numPieces; x++)
		assert(s.pieces[who][x-1] > s.pieces[who][x]);
	s.toMove = who;
	return true;
}

bool CCheckers::MovePlayerToGoal(CCState &state, int who)
{
	int cnt = 0;
	for (int y = 0; y < NUM_PIECES; y++)
		if (state.board[places[who][y]] == (1-who)+1)
			cnt++;
	if (cnt == NUM_PIECES)
		return false;
	for (int y = 0; y < NUM_PIECES; y++)
	{
		// erase old piece (if there)
		if (state.board[state.pieces[who][y]] == who+1)
			state.board[state.pieces[who][y]] = 0;
		// assign new piece
		state.pieces[who][y] = places[1-who][y];
		// check if overlap
		if (state.board[state.pieces[who][y]] != 0)
			return false;
		state.board[state.pieces[who][y]] = who+1;
	}
	state.toMove = 1-who;
	return true;
}

void CCheckers::Reset(CCState &state)
{
	state.toMove = 0;
	for (int x = 0; x < NUM_SPOTS; x++)
    {
		state.board[x] = 0;
    }
	for (int x = 0; x < NUM_PLAYERS; x++)
    {
		for (int y = 0; y < NUM_PIECES; y++)
		{
			state.pieces[x][NUM_PIECES-y-1] = places[x][y];
			state.board[state.pieces[x][NUM_PIECES-y-1]] = x+1;
		}
		local_sort(state.pieces[x]);
    }
	state.Verify();
	ResetGlobals();
}

/* Put P2 pieces in the P1 start
 * Change nothing else. Return false if P1 has
 * pieces in start already.
 */
bool CCheckers::SetP2Goal(CCState &s) const
{
	for (int x = 0; x < NUM_PIECES; x++)
	{
		s.pieces[1][x] = places[0][x];
		if (s.board[s.pieces[1][x]] == 1)
			return false;
		s.board[s.pieces[1][x]] = 2;
	}
	local_sort(s.pieces[1]);
	return true;
}


/**
 * Put p1 piece in the first possible position
 **/
void CCheckers::ResetP1Goal(CCState &s) const
{
	memset(s.board, 0, NUM_SPOTS*sizeof(int));
	for (int x = 0; x < NUM_PIECES; x++)
	{
		s.pieces[0][x] = g_places[1][x];
		s.board[s.pieces[0][x]] = 1;
	}
	s.toMove = 0;
	local_sort(s.pieces[0]);
}

/**
 * Put p1 piece in the first possible position
 **/
void CCheckers::ResetP1(CCState &state, int64_t &rank) const
{
	assert(NUM_PLAYERS == 2);
	assert(NUM_PIECES != 4);
	// Reset p1 pieces - works even if state is not valid for p1
	// (More robust code because we don't expect to call this often)
	for (int x = 0; x < NUM_SPOTS; x++)
	{
		if (state.board[x] == 1)
			state.board[x] = 0;
	}

	bool valid = true;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		state.pieces[0][x] = NUM_PIECES-x-1;
		if (state.board[NUM_PIECES-x-1] != 0)
			valid = false;
	}
	if (valid)
	{
		for (int x = 0; x < NUM_PIECES; x++)
			state.board[state.pieces[0][x]] = 1;
		rank = 0;
		return;
	}
	else {
		IncrementP1(state, rank);
	}

	int nextLoc = 0;
	for (int pieceNum = NUM_PIECES-1; pieceNum >= 0; pieceNum--)
	{
		int next = GetNextFreeLoc(state, nextLoc);
		state.pieces[0][pieceNum] = next;
		state.board[next] = 1;
		nextLoc = next+1;
	}
}

void CCheckers::ResetP1RelP2(CCState &state) const
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
}


int CCheckers::GetNextFreeLoc(CCState &s, int start) const
{
	for (int x = start; x < NUM_SPOTS; x++)
		if (s.board[x] == 0)
			return x;
	return -1;
}

/**
 * Move p1 pieces to next possible position
 **/
bool CCheckers::IncrementP1(CCState &state, int64_t &rank) const
{
	int nextLoc = -1;
	int pieceNum;
	for (pieceNum = 0; pieceNum < NUM_PIECES; pieceNum++)
	{
		nextLoc = GetNextFreeLoc(state, state.pieces[0][pieceNum]);
		if (nextLoc != -1)
		{
			break;
		}
	}
	if (nextLoc == -1)
		return false;
	for (int x = pieceNum; x >= 0; x--)
		state.board[state.pieces[0][x]] = 0;
	// set pieceNum & reset board
	for (; pieceNum >= 0; pieceNum--)
	{
		state.board[nextLoc] = 1;
		state.pieces[0][pieceNum] = nextLoc;
		// set all remaining pieces
		if (pieceNum != 0)
			nextLoc = GetNextFreeLoc(state, nextLoc+1);
	}
	return true;
}

/**
 * Move p1 pieces to next possible position
 **/
bool CCheckers::IncrementP1RelP2(CCState &state) const
{
	int nextLoc = -1;
	int pieceNum;
	for (pieceNum = 0; pieceNum < NUM_PIECES; pieceNum++)
	{
		nextLoc = GetNextFreeLoc(state, state.pieces[0][pieceNum]);
		if (nextLoc != -1)
		{
			break;
		}
	}
	if (nextLoc == -1)
		return false;
	for (int x = pieceNum; x >= 0; x--)
		state.board[state.pieces[0][x]] = 0;
	// set pieceNum & reset board
	for (; pieceNum >= 0; pieceNum--)
	{
		state.board[nextLoc] = 1;
		state.pieces[0][pieceNum] = nextLoc;
		// set all remaining pieces
		if (pieceNum != 0)
			nextLoc = GetNextFreeLoc(state, nextLoc+1);
	}
	return true;
}

void CCheckers::ResetGlobals()
{
	if (globalsReset == 0)
    {

		for (int x = 0; x < NUM_PLAYERS; x++)
			for (int y = 0; y < NUM_PIECES; y++)
				places[x][y] = g_places[x][y];
		for (int x = 0; x < ROWS+1; x++)
			count[x] = g_count[x];

		for (int x = 0; x < NUM_SPOTS; x++)
			mapx[x] = mapy[x] = -1;
		for (int x = 0; x < COLUMNS; x++)
			for (int y = 0; y < ROWS; y++)
				reverseMap[x][y] = -1;
		
		for (int x = 0; x < NUM_SPOTS; x++)
		{
			int a, b;
			realtoxy(x, a, b);
			//printf("%d = (%d, %d)\n", x, a, b);
			mapx[x] = a;
			mapy[x] = b;
			if (realfromxy(a, b) != x)
			{
				printf("Oh No! we made a mrstake! %d %d\n", a, b);
				exit(0);
			}
			reverseMap[a-1][b-1] = x;
		}
		for (int x = 0; x < NUM_SPOTS; x++)
		{
			for (int y = 0; y < NUM_SPOTS; y++)
			{
				distances[x][y] = newdistance(x, y);
				//				printf("%2d:%2d=%2d  ", x, y, distances[x][y]);
			}
			//printf("\n");
		}
		int loc = 0;
		for (int x = 0; x < NUM_SPOTS; x++)
		{
			if (leftHalf(x))
				leftID[x] = loc++;
			else {
				leftID[x] = leftID[flipHorizonally(x)];
			}
		}
		assert(NUM_LEFT_SIDE == loc);

		for (int y = 0; y < NUM_SPOTS; y++)
		{
			int row, col;
			int poss[6][2] =
			{{-1, -1}, {-1, 1}, {0, -2}, {0, 2}, {1, 1}, {1, -1}};
			
			torc(y, row, col);
			for (int x = 0; x < 6; x++) // 6 possible steps per pieces
			{
				stepLocations[y][x] = fromrc(row+poss[x][0], col+poss[x][1]);
				hopLocations[y][x] = fromrc(row+2*poss[x][0], col+2*poss[x][1]);
			}
		}
		//		int stepLocations[NUM_SPOTS][6];
		//		int hopLocations[NUM_SPOTS][6];

		globalsReset = 1;
    }
}

int CCheckers::getLeftID(int location)
{
	return leftID[location];
}

bool CCheckers::leftHalf(int location)
{
	int x1, y1, x2, y2;
	toxy(0, x1, y1);
	toxy(location, x2, y2);
	return (x2 <= x1);
}

bool CCheckers::rightHalf(int location)
{
	int x1, y1, x2, y2;
	toxy(0, x1, y1);
	toxy(location, x2, y2);
	return (x1 <= x2);
}

bool CCheckers::symmetricStart(CCState &s)
{
	int location1 = s.pieces[0][NUM_PIECES-1];
	int location2 = s.pieces[0][NUM_PIECES-2];
	int middle, x1, x2, y;
	toxy(0, middle, y);
//	printf("start location (0) at x: %d; ", middle);
	toxy(location1, x1, y);
	toxy(location2, x2, y);
//	printf("others at x: %d, %d", x1, x2);
//	printf("%s\n", ((x1 < middle) || ((x1 == middle) && (x2 <= middle)))?"not symmetric":"symmetric");
	return !((x1 < middle) || ((x1 == middle) && (x2 <= middle)));
}

bool CCheckers::specialStart(CCState &s)
{
	int location1 = s.pieces[0][NUM_PIECES-1];
	int location2 = s.pieces[0][NUM_PIECES-2];
	int middle, x1, x2, y;
	toxy(0, middle, y);
	//	printf("start location (0) at x: %d; ", middle);
	toxy(location1, x1, y);
	toxy(location2, x2, y);
	return (x1 >= middle || x2 >= middle);
}


int CCheckers::getCenterlineOffset(int location)
{
	int x, y;
	toxy(location, x, y);
	return count[y]/2;
}

int CCheckers::flipHorizonally(int location)
{
	int x1, y1, x2, y2;
	toxy(0, x1, y1);
	toxy(location, x2, y2);
	x2 = 2*x1-x2;
	return fromxy(x2, y2);
}

void CCheckers::setState(CCState &s, int p1[NUM_PIECES], int p2[NUM_PIECES])
{
	for (int x = 0; x < NUM_PIECES; x++)
	{
		s.board[s.pieces[0][x]] = 0;
		s.board[s.pieces[1][x]] = 0;
	}
	
	for (int x = 0; x < NUM_PIECES; x++) {
		s.board[p1[x]] = 1;
		s.pieces[0][x] = p1[x];
		s.board[p2[x]] = 2;
		s.pieces[1][x] = p2[x];
	}
}
/*
 void CCheckers::setState(CCState *c, int who)
 {
 int p[6];
 c->getValues(p);
 setState(p, who);
 }
 */
//void CCheckers::getPieces(int loc[NUM_PLAYERS][NUM_PIECES])
//{
//	for (int x = 0; x < NUM_PLAYERS; x++)
//		for (int y = 0; y < NUM_PIECES; y++)
//			loc[x][y] = pieces[x][y];
//}
//
//void CCheckers::getPieces(int loc[NUM_PIECES], int who)
//{
//	for (int y = 0; y < NUM_PIECES; y++)
//		loc[y] = pieces[who][y];
//	// sort them first!
//	for (int x = 0; x < NUM_PIECES; x++)
//		for (int y = 0; y < NUM_PIECES-1-x; y++)
//		{
//			if (loc[y] > loc[y+1])
//			{
//				int t = loc[y];
//				loc[y] = loc[y+1];
//				loc[y+1] = t;
//			}
//		}
//}

//void CCheckers::getBoard(int *b)
//{
//	for (int x = 0; x < NUM_SPOTS; x++)
//		b[x] = board[x];
//}

int CCheckers::getGoal(int who) const
{
	//return goals[NUM_PLAYERS-2][who];
	return who?0:(NUM_SPOTS-1);
}

int CCheckers::getStart(int who) const
{
	return who?(NUM_SPOTS-1):0;
}

int CCheckers::fromxy(int x, int y) const
{
	if ((x-1 >= 0) && (x-1 < COLUMNS) &&
		(y-1 >= 0) && (y-1 < ROWS))
		return reverseMap[x-1][y-1];
	return -1;
}

int CCheckers::realfromxy(int x, int y)
{
	if ((x < 1) || (y < 1) || (y > ROWS))
		return -1;
	if (((x%2 == 0) && (y%2 == 1)) ||
		((x%2 == 1) && (y%2 == 0)))
		return -1;
	
	int ans = 0, val, tmp;
	for (val = 0; val < y-1; val++)
		ans += count[val];
	tmp = (x-(13-count[val]));
	if (tmp < 0)
		return -1;
	tmp/=2;
	ans += tmp;
	if (tmp >= count[val])
		return -1;
	
	return ans;
}

int CCheckers::fromxytodiamond(int x, int y)
{
	if ((x < 1) || (y < 1) || (y > ROWS))
		return -1;
	if (((x%2 == 0) && (y%2 == 1)) ||
		((x%2 == 1) && (y%2 == 0)))
		return -1;
	int ans = 0, val, tmp;
	for (val = 0; val < y-1; val++)
		ans += count[val];
	tmp = (x-(13-count[val]));
	if (tmp < 0)
		return -1;
	tmp/=2;
	ans += tmp;
	if (tmp >= count[val])
		return -1;
	
	return ans;
}

int CCheckers::torc(int val, int &r, int &c) const
{
	return toxy(val, c, r);
}

int CCheckers::fromrc(int r, int c)
{
	return fromxy(c, r);
}

int CCheckers::toxy(int val, int &x, int &y) const
{
	if ((val < 0) || (val >= NUM_SPOTS))
	{
		x = 0; y = 0;
		return -1;
	}
	if (mapx[val] != -1)
    {
		x = mapx[val];
		y = mapy[val];
		return 0;
    }
	x = 0; y = 0;
	return -1;
}

int CCheckers::toxy_fast(int val, int &x, int &y) const
{
	x = mapx[val];
	y = mapy[val];
	return 0;
}

int CCheckers::realtoxy(int val, int &x, int &y)
{
	if ((val < 0) || (val >= NUM_SPOTS))
    {
		x = y = -1;
		return -1;
    }
	
	y = 0;
	while (1)
    {
		if (val < count[y])
			break;
		val = val-count[y];
		y++;
    }
	x = (13-count[y])+2*(val)+1;
	y++;
	
	return 0;
}

int CCheckers::toxydiamond(int val, int &x, int &y)
{
	if ((val < 0) || (val >= NUM_SPOTS))
    {
		x = y = -1;
		return -1;
    }
	
	y = 0;
	while (1)
    {
		if (val < count[y])
			break;
		val = val-count[y];
		y++;
    }
	x = (13-count[y])+2*(val)+1;
	y++;
	
	return 0;
}

bool CCheckers::Done(const CCState &s) const
{
	return (Winner(s) != -1);
//	//	//bool done = false;
//	for (int y = 0; y < NUM_PLAYERS; y++)
//	{
//		bool onePieceOurs = false;
//		for (int x = 0; x < NUM_PIECES; x++)
//		{
//			if (s.board[places[(1-y)][x]] == 0)
//				break;
//			if (s.board[places[(1-y)][x]] == y+1)
//				onePieceOurs = true;
//			if (x == NUM_PIECES-1)
//			{
////				if (done)
////					return false; // the state where both players are in the goal position is illegal...
////				done = true;
//				if (onePieceOurs)
//					return true;
//			}
//		}
//	}
//	//return done;
//	return false;
}

int CCheckers::Winner(const CCState &s) const
{
	for (int y = 0; y < NUM_PLAYERS; y++)
	{
		// Not true. Our opponent could move and then fill their own home area
		// with one of our pieces inside. (We disallow this in *our* rules)
		if (s.toMove == y) // to win, it must be our opponents turn to play
			continue;
		
		bool onePieceOurs = false;
		for (int x = 0; x < NUM_PIECES; x++)
		{
			if (s.board[places[(1-y)][x]] == 0)
				break;
			if (s.board[places[(1-y)][x]] == y+1)
				onePieceOurs = true;
			if (onePieceOurs && (x == NUM_PIECES-1))// && (s.toMove != y))
			{
				return y;
			}
		}
	}
	return -1;
}

bool CCheckers::Legal(const CCState &s) const
{
	// not legal if our home area is filled and its not our turn
	for (int y = 0; y < NUM_PLAYERS; y++)
	{
		int ours = 0;
		int count = 0;
		for (int x = 0; x < NUM_PIECES; x++)
		{
			if (s.board[places[(1-y)][x]] == 0)
				continue;
			else if (s.board[places[(1-y)][x]] == y+1)
				ours++;
			count++;
		}
		if (ours > 0 && (count == NUM_PIECES) && (s.toMove == y))
		{
//			if (s.toMove == y) // our turn to move
			return false;
		}
	}
	// 6 pieces in game, tip is empty
	if (NUM_PIECES == 6 && NUM_PLAYERS == 2 &&
		s.board[NUM_SPOTS-1] == 0 && s.board[NUM_SPOTS-2] == 2 && s.board[NUM_SPOTS-3] == 2 && s.board[NUM_SPOTS-4] == 2 && s.board[NUM_SPOTS-6] == 2)
	{
		//s.PrintASCII();
		return false;
	}
	if (NUM_PIECES == 6 && NUM_PLAYERS == 2 &&
		s.board[0] == 0 && s.board[1] == 1 && s.board[2] == 1 && s.board[3] == 1 && s.board[5] == 1)
	{
		//s.PrintASCII();
		return false;
	}
	return true;
}

int CCheckers::GetNumPiecesInGoal(const CCState &s, int who) const
{
	int cnt = 0;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		if (s.board[places[(1-who)][x]] == who+1)
			cnt++;
	}
	return cnt;
}

int CCheckers::GetNumPiecesInStart(const CCState &s, int who) const
{
	int cnt = 0;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		if (s.board[places[who][x]] == who+1)
			cnt++;
	}
	return cnt;
}

void CCheckers::ApplyMove(CCState &s, CCMove *m)
{	
//	s.Verify();
	if (m->from == m->to) // no move [pass, my pieces are stuck]
		return;

	s.board[m->to] = s.board[m->from];
	s.board[m->from] = 0;

	if (m->from < m->to)
	{
		s.pieces[s.toMove][m->which] = m->to;
		for (int x = m->which-1; x >= 0; x--)
		{
			if (s.pieces[s.toMove][x] < m->to)
			{
				s.pieces[s.toMove][x+1] = s.pieces[s.toMove][x];
				s.pieces[s.toMove][x] = m->to;
			}
			else break;
		}
	}
	else {
		s.pieces[s.toMove][m->which] = m->to;
		for (int x = m->which+1; x < NUM_PIECES; x++)
		{
			if (s.pieces[s.toMove][x] > m->to)
			{
				s.pieces[s.toMove][x-1] = s.pieces[s.toMove][x];
				s.pieces[s.toMove][x] = m->to;
			}
			else break;
		}
	}
	s.toMove = (s.toMove+1)%NUM_PLAYERS;
//	s.Verify();
}

void CCheckers::UndoMove(CCState &s, CCMove *m)
{
//	s.Verify();
	s.toMove = (s.toMove+NUM_PLAYERS-1)%NUM_PLAYERS;
	
	if (m->from == m->to) // no move
		return;
	
	s.board[m->from] = s.board[m->to];
	s.board[m->to] = 0;
	
	if (m->from < m->to)
	{
		int assign = m->from;
		for (int x = m->which; x >= 0; x--)
		{
			if (s.pieces[s.toMove][x] == m->to)
			{
				s.pieces[s.toMove][x] = assign;
				break;
			}
			int tmp = s.pieces[s.toMove][x];
			s.pieces[s.toMove][x] = assign;
			assign = tmp;
		}
	}
	else {
		int assign = m->from;
		for (int x = m->which; x < NUM_PIECES; x++)
		{
			if (s.pieces[s.toMove][x] == m->to)
			{
				s.pieces[s.toMove][x] = assign;
				break;
			}
			int tmp = s.pieces[s.toMove][x];
			s.pieces[s.toMove][x] = assign;
			assign = tmp;
		}
	}
//	s.Verify();
}

void CCheckers::ApplyReverseMove(CCState &s, CCMove *m)
{
//	s.Verify();
	s.toMove = (s.toMove+NUM_PLAYERS-1)%NUM_PLAYERS;
	
	if (m->from == m->to) // no move
		return;
	
	s.board[m->from] = s.board[m->to];
	s.board[m->to] = 0;
	
	if (m->from > m->to)
	{
		s.pieces[s.toMove][m->which] = m->from;
		for (int x = m->which-1; x >= 0; x--)
		{
			if (s.pieces[s.toMove][x] < m->from)
			{
				s.pieces[s.toMove][x+1] = s.pieces[s.toMove][x];
				s.pieces[s.toMove][x] = m->from;
			}
			else break;
		}
	}
	else {
		s.pieces[s.toMove][m->which] = m->from;
		for (int x = m->which+1; x < NUM_PIECES; x++)
		{
			if (s.pieces[s.toMove][x] > m->from)
			{
				s.pieces[s.toMove][x-1] = s.pieces[s.toMove][x];
				s.pieces[s.toMove][x] = m->from;
			}
			else break;
		}
	}

//	s.Verify();
}

void CCheckers::UndoReverseMove(CCState &s, CCMove *m)
{
//	s.Verify();
	if (m->from == m->to) // no move [pass, my pieces are stuck]
		return;
	
	s.board[m->to] = s.board[m->from];
	s.board[m->from] = 0;
	
	if (m->from > m->to)
	{
		int assign = m->to;
		for (int x = m->which; x >= 0; x--)
		{
			if (s.pieces[s.toMove][x] == m->from)
			{
				s.pieces[s.toMove][x] = assign;
				break;
			}
			int tmp = s.pieces[s.toMove][x];
			s.pieces[s.toMove][x] = assign;
			assign = tmp;
		}
	}
	else {
		int assign = m->to;
		for (int x = m->which; x < NUM_PIECES; x++)
		{
			if (s.pieces[s.toMove][x] == m->from)
			{
				s.pieces[s.toMove][x] = assign;
				break;
			}
			int tmp = s.pieces[s.toMove][x];
			s.pieces[s.toMove][x] = assign;
			assign = tmp;
		}
	}

	s.toMove = (s.toMove+1)%NUM_PLAYERS;
//	s.Verify();
}

CCMove *CCheckers::getMovesForward(const CCState &s) const
{
	CCMove *m = getMoves(s);
	CCMove *result = 0;
	
	while (m)
	{
		if (distance(m->from, getGoal(s.toMove)) >= distance(m->to, getGoal(s.toMove)))
		{
			CCMove *tmp = m->next;
			m->next = result;
			result = m;
			m = tmp;
		}
		else {
			CCMove *tmp = m->next;
			m->next = 0;
			freeMove(m);
			m = tmp;
		}
	}
	return result;
}


CCMove *CCheckers::getMoves(const CCState &s) const
{
	int who;
	CCMove ans(0, 0, 0);
	CCMove *res;
	who = s.toMove;// getNextPlayerNum();
	// for some variation, random switch the order in which we consider moves
	// which will just switch certain ties around in the tree
	if (0)//(0 == random()%2)
	{
		for (int x = NUM_PIECES-1; x >= 0; x--)
		{
			forbidden1 = 0;
			forbidden2 = 0;
			getHops(who, x, s.pieces[who][x], s.pieces[who][x], s.board, &ans, true);
		}
		for (int x = NUM_PIECES-1; x >= 0; x--)
		{
			CCMove *t;
			res = getSteps(who, x, s.pieces[who][x], s.board, true);
			while (res)
			{
				t = res->next;
				res->next = ans.next;
				ans.next = res;
				res = t;
			}
		}
	}
	else {
		{
			for (int x = 0; x < NUM_PIECES; x++)
			{
				forbidden1 = 0;
				forbidden2 = 0;
				getHops(who, x, s.pieces[who][x], s.pieces[who][x], s.board, &ans, true);
			}
			for (int x = 0; x < NUM_PIECES; x++)
			{
				CCMove *t;
				res = getSteps(who, x, s.pieces[who][x], s.board, true);
				while (res)
				{
					t = res->next;
					res->next = ans.next;
					ans.next = res;
					res = t;
				}
			}
		}
	}
	res = ans.next;
	ans.next = 0;
	
	if (res == 0)
    {
		s.Print();
		printf("NO MOVES GENERATED FORWARD!!! [%d]\n", who);
		//exit(0);
		//return 0;
		// generate a pass move, because all pieces are trapped
		CCMove *m = getNewMove();
		m->from = s.pieces[who][0];
		m->to = s.pieces[who][0];
		return m;
    }
	return res;
}

CCMove *CCheckers::getMovesForPiece(CCState &s, int which)
{
	int who;
	CCMove ans(0, 0, 0);
	CCMove *res;
	who = s.toMove;// getNextPlayerNum();
	// for some variation, random switch the order in which we consider moves
	// which will just switch certain ties around in the tree
	{
		//for (int x = NUM_PIECES-1; x >= 0; x--)
		{
			int x = which;
			//			printf("ERASE\n");
			//			forbidden.erase(forbidden.begin(), forbidden.end());
			forbidden1 = 0;
			forbidden2 = 0;
			//			forbidden.resize(0);
			//			forbidden.resize(NUM_SPOTS);
			getHops(who, x, s.pieces[who][x], s.pieces[who][x], s.board, &ans, true);
		}
		//for (int x = NUM_PIECES-1; x >= 0; x--)
		{
			int x = which;
			CCMove *t;
			res = getSteps(who, x, s.pieces[who][x], s.board, true);
			while (res)
			{
				t = res->next;
				res->next = ans.next;
				ans.next = res;
				//ans.add(res);
				res = t;
			}
		}
	}
	res = ans.next;
	ans.next = 0;
	return res;
}


CCMove *CCheckers::getReverseMoves(CCState &s)
{
	int who;
	CCMove ans(0, 0, 0);
	CCMove *res;
	who = 1-s.toMove;// getNextPlayerNum();
	// for some variation, random switch the order in which we consider moves
	// which will just switch certain ties around in the tree
	{
		for (int x = NUM_PIECES-1; x >= 0; x--)
		{
//			printf("ERASE\n");
//			forbidden.erase(forbidden.begin(), forbidden.end());
			forbidden1 = 0;
			forbidden2 = 0;
//			forbidden.resize(0);
//			forbidden.resize(NUM_SPOTS);
			getHops(who, x, s.pieces[who][x], s.pieces[who][x], s.board, &ans, false);
		}
		for (int x = NUM_PIECES-1; x >= 0; x--)
		{
			CCMove *t;
			res = getSteps(who, x, s.pieces[who][x], s.board, false);
			while (res)
			{
				t = res->next;
				//res->next = 0;
				res->next = ans.next;
				ans.next = res;
				//ans.add(res);
				res = t;
			}
		}
	}
	res = ans.next;
	ans.next = 0;
	
	if (res == 0)
    {
		s.Print();
		s.PrintASCII();
		printf("NO MOVES GENERATED BACKWARD!!! [%d]\n", who);
		//exit(0);
		//return 0;
		// generate a pass move, because all pieces are trapped
		CCMove *m = getNewMove();
		m->from = s.pieces[who][0];
		m->to = s.pieces[who][0];
		return m;
    }
	for (CCMove *t = res; t; t = t->next)
	{
		int tmp = t->from;
		t->from = t->to;
		t->to = tmp;
	}
	return res;
}


CCMove *CCheckers::getSteps(int who, int where, int pos, const int *board, bool forward) const
{
	int row, col, val;
//	int np = NUM_PLAYERS;
//	int poss[6][2] =
//    {{-1, -1}, {-1, 1}, {0, -2}, {0, 2}, {1, 1}, {1, -1}};
	CCMove ans(0,0,0), *res;
	
	torc(pos, row, col);
	for (int x = 0; x < 6; x++) // 6 possible steps per pieces
    {
		// if it is legal and there is no piece there
		val = stepLocations[pos][x];
		//		if (((val = fromrc(row+poss[x][0], col+poss[x][1])) != -1) &&
		//			(board[val] == 0))
		if (val != -1 && board[val] == 0)
		{
			int legal = 1;
			if (legal) {
				CCMove *m = getNewMove();
				m->from = pos; m->to = val;
				m->which = where;
				m->next = ans.next;
				ans.next = m;
			}
			else {
			}
		}
		else if (pos == 0)
		{
		}
    }
	res = (CCMove*)ans.next;
	ans.next = 0;
	return res;
}


void CCheckers::getHops(int who, int where, int origPos, int pos, const int *board, CCMove *m, bool forward) const
{
	int row, col;
	int x, val1, val2;
	toxy(pos, col, row);
	for (x = 0; x < 6; x++) // 6 possible hops
    {
		val1 = stepLocations[pos][x];
		val2 = hopLocations[pos][x];
		if (val1 != -1 && board[val1] && val2 != -1 && !board[val2])
		{
			int legal = 1;
			if (legal && (((val2 < 64) && !(forbidden1&(1ull<<val2))) ||
						  ((val2 >= 64) && !(forbidden2&(1ull<<(val2-64))))))
			{
				CCMove *mm = getNewMove();
				mm->from = origPos; mm->to = val2;
				mm->which = where;
				if (val2 < 64)
					forbidden1 |= (1ull<<val2);
				else
					forbidden2 |= (1ull<<(val2-64));
				mm->next = m->next;
				m->next = mm;
				getHops(who, where, origPos, val2, board, m, forward);
			}
		}
    }
}

//int CCheckers::getNextPlayerNum() const
//{
//	return toMove;
//}
//
//int CCheckers::getPreviousPlayerNum() const
//{
//	return (toMove+NUM_PLAYERS-1)%NUM_PLAYERS;
//}

int CCheckers::startDistance(CCState &s, int who) const
{
	int sum = 0;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		sum += distance(s.pieces[who][x], getGoal(1-who));
	}
	return sum;
}

int CCheckers::goalDistance(CCState &s, int who) const
{
	int sum = 0;
	for (int x = 0; x < NUM_PIECES; x++)
	{
		sum += distance(s.pieces[who][x], getGoal(who));		
	}
	return sum;
}

// this is the number of hops to our goal
int CCheckers::distance(int x1, int y1, int x2, int y2) const
{
	int val;
	
	val = abs(y1-y2);
	if (val >= abs(x1-x2))
		return val;
	return val+abs(abs(x1-x2)-val)/2;
}

int CCheckers::newdistance(int v1, int v2)
{
	int x1, x2, y1, y2;
	
	CCheckers::toxy(v1, x1, y1);
	CCheckers::toxy(v2, x2, y2);
	return CCheckers::distance(x1, y1, x2, y2);
	/*
	 int x1, x2, y1, y2;
	 int newx1, newx2;
	 int diffx, diffy;
	 
	 CCheckers::toxy(v1, x1, y1);
	 CCheckers::toxy(v2, x2, y2);
	 newx1 = (x1-y1+9)/2;
	 newx2 = (x1-y2+9)/2;
	 
	 diffx = newx1-newx2;
	 diffy = y1-y2;
	 return (abs(diffx)+abs(diffy)+abs(diffx-diffy))/2;*/
}

// this is the number of hops from v1 to v2 goal
int CCheckers::distance(int v1, int v2) const
{
	return distances[v1][v2];
	/*
	 int x1, x2, y1, y2;
	 
	 CCheckers::toxy(v1, x1, y1);
	 CCheckers::toxy(v2, x2, y2);
	 return CCheckers::distance(x1, y1, x2, y2);*/
}

// this is the offset from the center line towards goal
int CCheckers::lineoffset(int x1, int y1, int x2, int y2, int x3, int y3)
{
	//int xoff, yoff;
	int base = distance(x2, y2, x3, y3);
	return distance(x1, y1, x2, y2) + distance(x1, y1, x3, y3) - base;
	/*
	 while (1) {
	 if ((x2 >= 7) && (x2 <= ROWS))
	 return (abs(x1-x2)+1)/2;
	 rotate60clockwise(x1, y1);
	 rotate60clockwise(x2, y2);
	 }
	 
	 return -1; */
}

void CCheckers::rotate60clockwise(int &x, int &y)
{
	//int dist = distance(x, y, 36); // distance to center
	int dist = distance(x, y, 10, 7); // distance to center
	
	for (int t = 0; t < dist; t++) // move dist units around center
    {
		switch (getQuad(x, y)) {
			case 0: x+=2; //printf(" 0");
				break;
			case 1: x+=1; y+=1; //printf(" 1");
				break;
			case 2: x-=1; y+=1; //printf(" 2");
				break;
			case 3: x-=2; //printf(" 3");
				break;
			case 4: x-=1; y-=1; //printf(" 4");
				break;
			case 5: x+=1; y-=1; //printf(" 5");
				break;
			default: // center piece
				break;
		}
    }
}

int CCheckers::rotate60clockwise(int pos)
{
	int x, y;
	
	toxy(pos, x, y);
	//printf("Rotating %d / (%d, %d) dist %d\n", pos, x, y, dist);
	rotate60clockwise(x, y);
	//printf(" New Position: %d, (%d, %d)\n", fromxy(x, y), x, y);
	return fromxy(x, y);
}

int CCheckers::getQuad(int x, int y)
{
	if ((y <= 7) && (x-y <= 1))
		return 5;
	if ((y <= 7) && (x-y >= 3) && (x+y <= 15))
		return 0;
	if ((y < 7) && (x+y >= 17))
		return 1;
	if ((y >= 7) && (x-y >= 5))
		return 2;
	if ((y > 7) && (x-y <= 3) && (x+y >= 19))
		return 3;
	if ((y > 7) && (x+y <= 17))
		return 4;
	//printf("No Quadrant!!! %d %d\n", x, y);
	return -1;
}

// this is the offset from the center line towards goal
int CCheckers::goalOffset(int pos, int plyr)
{
	int x1, x2, x3, y1, y2, y3;
	
	torc(pos, y1, x1);
	torc(getGoal(plyr), y2, x2);
	torc(getStart(plyr), y3, x3);
	
	return lineoffset(x1, y1, x2, y2, x3, y3);
}


CCMove *CCheckers::allocateMoreMoves(int n)
{
	CCMove *m = 0;
	for (int x = 0; x < n; x++)
		m = new CCMove(m);
	return m;
}

int CCMove::add(CCMove *m)
{
	if ((m == 0) || (this == 0))
		return 0;
	if ((m->from == from) && (m->to == to)) {
		//printf("In Already: %d %d\n", from, to);
		return 0;
	}
	if (next == 0)
    {
		next = m;
		return 1;
    }
	return ((CCMove*)next)->add(m);
}

CCMove *CCMove::clone(CCheckers &cc) const
{
	CCMove *m = cc.getNewMove();
	m->from = from; m->to = to; m->next = 0;
	m->which = which;
	return m;
}

double CCheckers::eval(const CCState &s, int who)
{
//	int pos[NUM_PIECES];
//	getPieces(pos, who);
	
	int maneval = 0;
	int maxdist = 0;
	int mindist = 100;
	int distScore = 0;
	int tot=0, moveCount = 0;
	//int goal = getGoal(who);
	/*
	 Move *t, *m = p->getMoves();
	 for (t = m; t != 0; t=t->next)
		if (t->dist > 0)
	 moveCount++;
		else
	 break;
	 delete m;
	 */
	for (int x = 0; x < NUM_PIECES; x++) {
		//		if (pos[x] == -1)
		//			exit(0);
//		distScore = distance(s.pieces[who], goal);
		if (distScore > maxdist)
			maxdist = distScore;
		if (distScore < mindist)
			mindist = distScore;
		if (distScore < 4)
			tot++;
		maneval += distScore*distScore;
	}
	
	// win or loss
	if (tot == NUM_PIECES) {
		// don't need to subtract because we are doing an iterative search
		return 1000; // -g->getDepth();
	}
	return (double)(- maneval + tot*10 -
					(maxdist-mindist) + moveCount*moveCount);
}


