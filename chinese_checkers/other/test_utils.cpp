#include <iostream>
#include "CCheckers.h"
#include "test_utils.h"

using namespace std;

ostream& operator<<(ostream& out, const int* board)
{
  for(int i=0; i<NUM_SPOTS; ++i){
    if(i > 0){
      out << " ";
    }
    out << board[i];
  }
  return out;
}

